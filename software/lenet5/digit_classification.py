"""
PYNQ-Z2 end-to-end digit recognition + latency analysis
========================================================

Current hardware convention
---------------------------
- Accelerator IP instance: CNNaccelerator_v1_0_0
- Accelerator base:        0x40000000
- AXI DMA instance:        axi_dma_0
- DMA base:                0x40400000
- Useful convolution beats: 254*254 = 64516
- AXI packet beats:         64517
- Final beat is discarded (known zero tail beat)

Pipeline
--------
Input image
 -> Conv1 on FPGA (6 parallel filters)
 -> MaxPool1 on ARM
 -> Conv2 on FPGA (16 output filters, 16 passes)
 -> MaxPool2 on ARM
 -> FC1/FC2/FC3 on ARM
 -> digit 0..9

The script ALSO computes a software quantized reference for the same image.
This is important because the current Conv2 hardware has shown non-zero
differences from the golden reference on some filters.

Python 3.6 compatible.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pynq import Overlay, allocate

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# ============================================================================
# USER SETTINGS
# ============================================================================

# Put your image in the same Jupyter folder and change this if needed.
# Supported:
#   digit.png / digit.jpg / digit.jpeg / digit.bmp
#   test_image.npy
INPUT_PATH = "digit.png"

# For ordinary white-paper / black-digit images, AUTO_INVERT=True converts
# them to MNIST style (bright digit on dark background).
AUTO_INVERT = True

# If INPUT_PATH does not exist but test_image.npy exists, the script will use it.
FALLBACK_NPY = "test_image.npy"

# Show simple plots at the end.
SHOW_PLOTS = True


# ============================================================================
# HARDWARE CONSTANTS
# ============================================================================

ACCEL_IP_NAME = "CNNaccelerator_v1_0_0"
DMA_IP_NAME   = "axi_dma_0"

ACCEL_BASE_EXPECTED  = 0x40000000
ACCEL_RANGE_EXPECTED = 0x00001000
DMA_BASE_EXPECTED    = 0x40400000
DMA_RANGE_EXPECTED   = 0x00010000

NUM_IN_BEATS      = 256 * 256
USEFUL_OUT_BEATS  = 254 * 254          # 64516
PACKET_OUT_BEATS  = USEFUL_OUT_BEATS + 1   # 64517

MODE_ACCUMULATE         = 0
MODE_BYPASS_RAW         = 1
MODE_BYPASS_QUANT       = 2
MODE_BYPASS_RELU_QUANT  = 3

REG_CONTROL  = 0x00
REG_STATUS   = 0x04
REG_CONFIG   = 0x08
REG_ACC_BIAS = 0x0C
REG_VE_BIAS0 = 0x10
REG_WEIGHTS0 = 0x28

DMA_TIMEOUT_S = 2.0
POLL_S = 0.0002


def u32(x):
    return int(x) & 0xFFFFFFFF


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def resize_2d_uint8(arr, out_hw):
    """
    Resize a uint8 2-D image using Pillow if available.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError(
            "Pillow is required for PNG/JPG preprocessing. "
            "Use test_image.npy instead or install Pillow."
        )
    im = Image.fromarray(arr.astype(np.uint8), mode="L")
    im = im.resize((out_hw[1], out_hw[0]), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def mnist_like_preprocess(path):
    """
    Returns a 28x28 uint8 image.

    For PNG/JPG:
      1. grayscale
      2. optional automatic inversion
      3. remove weak background
      4. crop around digit
      5. resize digit to fit inside ~20x20
      6. center inside 28x28

    For .npy:
      expects 784 values / 28x28 and only normalizes type/range.
    """
    if path.lower().endswith(".npy"):
        x = np.load(path)
        x = np.asarray(x)

        if x.size != 28 * 28:
            raise RuntimeError(
                "%s must contain exactly 784 values; got shape %s"
                % (path, x.shape)
            )

        x = x.reshape(28, 28)

        if np.issubdtype(x.dtype, np.floating):
            if float(x.max()) <= 1.5:
                x = np.rint(x * 255.0)
            else:
                x = np.rint(x)

        return np.clip(x, 0, 255).astype(np.uint8)

    if not PIL_AVAILABLE:
        raise RuntimeError(
            "Pillow is not available. Use a 28x28 test_image.npy instead."
        )

    im = Image.open(path).convert("L")
    x = np.asarray(im, dtype=np.uint8)

    # Convert typical black-on-white handwritten image to MNIST polarity.
    if AUTO_INVERT and float(np.mean(x)) > 127.0:
        x = 255 - x

    # Estimate background and suppress low-level noise.
    # This deliberately remains simple and deterministic.
    threshold = max(20, int(0.12 * int(x.max())))
    mask = x > threshold

    if not np.any(mask):
        raise RuntimeError("Could not find a visible digit in %s" % path)

    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1

    crop = x[r0:r1, c0:c1]

    # Preserve aspect ratio and fit the digit inside 20x20, similar to MNIST.
    h, w = crop.shape
    scale = min(20.0 / float(h), 20.0 / float(w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    crop = resize_2d_uint8(crop, (nh, nw))

    out = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - nh) // 2
    left = (28 - nw) // 2
    out[top:top + nh, left:left + nw] = crop

    return out


# ============================================================================
# MODEL / MATH HELPERS
# ============================================================================

def pack_6x3x3_to_regs(weights_6x3x3):
    w = np.asarray(weights_6x3x3, dtype=np.int8).reshape(6, 3, 3)
    raw = w.tobytes() + b"\x00\x00"
    return np.frombuffer(raw, dtype=np.uint32)


def maxpool2x2(x):
    C, H, W = x.shape
    H2 = (H // 2) * 2
    W2 = (W // 2) * 2
    x = x[:, :H2, :W2]
    return x.reshape(C, H2 // 2, 2, W2 // 2, 2).max(axis=(2, 4)).astype(np.uint8)


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def conv2d_int_golden(pixels, weights, bias, shift):
    Cout, Cin, K, _ = weights.shape
    H = pixels.shape[1]
    W = pixels.shape[2]
    Ho = H - K + 1
    Wo = W - K + 1

    out = np.zeros((Cout, Ho, Wo), dtype=np.int64)

    for co in range(Cout):
        acc = np.zeros((Ho, Wo), dtype=np.int64)

        for ci in range(Cin):
            for ky in range(K):
                for kx in range(K):
                    acc += (
                        pixels[ci,
                               ky:ky + Ho,
                               kx:kx + Wo].astype(np.int64)
                        * int(weights[co, ci, ky, kx])
                    )

        acc += int(bias[co])
        acc = np.where(acc < 0, 0, acc)
        acc = acc >> int(shift)
        out[co] = np.clip(acc, 0, 255)

    return out.astype(np.uint8)


def fc_forward(pool2, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
    feat = pool2.astype(np.float32).reshape(400)

    x1 = np.dot(fc1_w, feat) + fc1_b
    x1 = np.maximum(x1, 0)

    x2 = np.dot(fc2_w, x1) + fc2_b
    x2 = np.maximum(x2, 0)

    logits = np.dot(fc3_w, x2) + fc3_b
    probs = softmax(logits)

    pred = int(np.argmax(probs))
    confidence = float(probs[pred] * 100.0)

    return logits, probs, pred, confidence


# ============================================================================
# LOAD OVERLAY
# ============================================================================

BITSTREAM_CANDIDATES = ["CNNaccelerator.bit", "design_1.bit"]

bitstream_name = None
for candidate in BITSTREAM_CANDIDATES:
    if os.path.exists(candidate):
        bitstream_name = candidate
        break

if bitstream_name is None:
    raise FileNotFoundError(
        "Could not find CNNaccelerator.bit or design_1.bit."
    )

print("=" * 80)
print("PYNQ END-TO-END DIGIT RECOGNITION")
print("=" * 80)
print("Loading overlay:", bitstream_name)

overlay = Overlay(bitstream_name)

if ACCEL_IP_NAME not in overlay.ip_dict:
    raise RuntimeError(
        "Expected accelerator '%s' not found. Available: %s"
        % (ACCEL_IP_NAME, sorted(overlay.ip_dict.keys()))
    )

if DMA_IP_NAME not in overlay.ip_dict:
    raise RuntimeError(
        "Expected DMA '%s' not found. Available: %s"
        % (DMA_IP_NAME, sorted(overlay.ip_dict.keys()))
    )

acc_info = overlay.ip_dict[ACCEL_IP_NAME]
dma_info = overlay.ip_dict[DMA_IP_NAME]

acc_base = int(acc_info.get("phys_addr", -1))
acc_range = int(acc_info.get("addr_range", -1))
dma_base = int(dma_info.get("phys_addr", -1))
dma_range = int(dma_info.get("addr_range", -1))

print("Accelerator: base=0x%08X range=0x%X" % (acc_base, acc_range))
print("DMA        : base=0x%08X range=0x%X" % (dma_base, dma_range))

if acc_base != ACCEL_BASE_EXPECTED or acc_range != ACCEL_RANGE_EXPECTED:
    raise RuntimeError("Accelerator address map mismatch.")

if dma_base != DMA_BASE_EXPECTED or dma_range != DMA_RANGE_EXPECTED:
    raise RuntimeError("DMA address map mismatch.")

engine = getattr(overlay, ACCEL_IP_NAME)
dma = getattr(overlay, DMA_IP_NAME)

print("Address map: VERIFIED")
print("AXI packet : %d useful + 1 discarded tail = %d beats"
      % (USEFUL_OUT_BEATS, PACKET_OUT_BEATS))


# ============================================================================
# LOAD MODEL FILES
# ============================================================================

conv1_w = np.load("conv1_weights_int8.npy").astype(np.int8)
conv1_b = np.load("conv1_bias_int32.npy").astype(np.int32)

conv2_w = np.load("conv2_weights_int8.npy").astype(np.int8)
conv2_b = np.load("conv2_bias_int32.npy").astype(np.int32)

shift1, shift2 = [int(x) for x in np.load("shift_amounts.npy")]

fc1_w = np.load("fc1_weights.npy").astype(np.float32)
fc1_b = np.load("fc1_bias.npy").astype(np.float32)
fc2_w = np.load("fc2_weights.npy").astype(np.float32)
fc2_b = np.load("fc2_bias.npy").astype(np.float32)
fc3_w = np.load("fc3_weights.npy").astype(np.float32)
fc3_b = np.load("fc3_bias.npy").astype(np.float32)

assert conv1_w.shape == (6, 1, 3, 3)
assert conv2_w.shape == (16, 6, 3, 3)
assert fc1_w.shape == (120, 400)
assert fc2_w.shape == (84, 120)
assert fc3_w.shape == (10, 84)

conv1_words = pack_6x3x3_to_regs(conv1_w[:, 0])
conv2_words = [pack_6x3x3_to_regs(conv2_w[f]) for f in range(16)]

print("Model files: LOADED")
print("shift1=%d shift2=%d" % (shift1, shift2))


# ============================================================================
# INPUT IMAGE
# ============================================================================

input_path = INPUT_PATH

if not os.path.exists(input_path):
    if os.path.exists(FALLBACK_NPY):
        input_path = FALLBACK_NPY
        print("Input '%s' not found; using '%s'." % (INPUT_PATH, input_path))
    else:
        raise FileNotFoundError(
            "Put '%s' (or '%s') in this Jupyter folder."
            % (INPUT_PATH, FALLBACK_NPY)
        )

t0 = time.perf_counter()
test_img = mnist_like_preprocess(input_path)
preprocess_ms = (time.perf_counter() - t0) * 1000.0

print("\nInput:", input_path)
print("Preprocessed image:", test_img.shape, test_img.dtype,
      "min=%d max=%d" % (int(test_img.min()), int(test_img.max())))
print("Preprocess time: %.3f ms" % preprocess_ms)


# ============================================================================
# HARDWARE HELPERS
# ============================================================================

def soft_reset():
    engine.write(REG_CONTROL, 0x02)
    time.sleep(0.0002)
    engine.write(REG_CONTROL, 0x00)


def write_config(shift, mode):
    cfg = (
        ((PACKET_OUT_BEATS & 0xFFFF) << 16)
        | ((int(shift) & 0x1F) << 2)
        | (mode & 0x3)
    )
    engine.write(REG_CONFIG, u32(cfg))


def read_dmasr(channel):
    return int(channel._mmio.read(channel._offset + 4))


def read_dma_length(channel):
    return int(channel._mmio.read(channel._offset + 0x28))


def launch_dma(in_buf, out_buf):
    """
    Run one accelerator transaction.

    Returns:
      dma_ms, received_beats

    Timing begins immediately before programming S2MM and ends after both
    DMA directions become idle. There is NO artificial 1 ms / 5 ms delay.
    """
    engine.write(REG_CONTROL, 0x01)

    status = int(engine.read(REG_STATUS))
    ready = (status >> 1) & 1

    if not ready:
        engine.write(REG_CONTROL, 0x00)
        raise RuntimeError("Accelerator not ready before DMA launch.")

    t0 = time.perf_counter()

    # Program S2MM first, then MM2S. transfer() itself is synchronous MMIO
    # programming, so no diagnostic sleep is inserted here.
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)

    while True:
        send_stat = read_dmasr(dma.sendchannel)
        recv_stat = read_dmasr(dma.recvchannel)

        if send_stat & 0x70:
            engine.write(REG_CONTROL, 0x00)
            raise RuntimeError(
                "MM2S DMA error: 0x%08X" % send_stat
            )

        if recv_stat & 0x70:
            engine.write(REG_CONTROL, 0x00)
            raise RuntimeError(
                "S2MM DMA error: 0x%08X" % recv_stat
            )

        if (send_stat & 0x2) and (recv_stat & 0x2):
            break

        if (time.perf_counter() - t0) > DMA_TIMEOUT_S:
            engine.write(REG_CONTROL, 0x00)
            raise RuntimeError(
                "DMA timeout. MM2S=0x%08X S2MM=0x%08X"
                % (send_stat, recv_stat)
            )

        time.sleep(POLL_S)

    t1 = time.perf_counter()

    dma.sendchannel.wait()
    dma.recvchannel.wait()

    try:
        out_buf.invalidate()
    except Exception:
        pass

    received_bytes = read_dma_length(dma.recvchannel)
    received_beats = received_bytes // 8

    engine.write(REG_CONTROL, 0x00)

    if received_beats != PACKET_OUT_BEATS:
        raise RuntimeError(
            "Received %d beats; expected exactly %d."
            % (received_beats, PACKET_OUT_BEATS)
        )

    return (t1 - t0) * 1000.0, received_beats


# ============================================================================
# ALLOCATE DMA BUFFERS ONCE
# ============================================================================

in_buf = allocate(shape=(NUM_IN_BEATS, 8), dtype=np.uint8)
out_buf = allocate(shape=(PACKET_OUT_BEATS, 8), dtype=np.uint8)

try:
    print("\n" + "=" * 80)
    print("HARDWARE-ASSISTED INFERENCE")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Inference wall clock starts here.
    # Excludes overlay download, model-file loading and image-file disk I/O.
    # Includes buffer preparation, MMIO setup, DMA, pooling and FC inference.
    # ------------------------------------------------------------------------
    inference_t0 = time.perf_counter()

    # ========================================================================
    # CONV1
    # ========================================================================

    stage_t0 = time.perf_counter()

    image_256 = np.zeros((256, 256), dtype=np.uint8)
    image_256[:28, :28] = test_img
    flat = image_256.reshape(-1)

    in_buf[:] = 0
    for ch in range(6):
        in_buf[:, ch] = flat

    out_buf[:] = 0

    soft_reset()
    write_config(shift1, MODE_BYPASS_RELU_QUANT)

    engine.write(REG_ACC_BIAS, 0)

    for e in range(6):
        engine.write(REG_VE_BIAS0 + 4 * e, u32(conv1_b[e]))

    for r in range(14):
        engine.write(REG_WEIGHTS0 + 4 * r, int(conv1_words[r]))

    conv1_dma_ms, conv1_packet_beats = launch_dma(in_buf, out_buf)

    conv1_out = np.zeros((6, 26, 26), dtype=np.uint8)

    for ch in range(6):
        full_map = np.asarray(
            out_buf[:USEFUL_OUT_BEATS, ch]
        ).reshape(254, 254)
        conv1_out[ch] = full_map[:26, :26]

    conv1_tail = np.asarray(out_buf[USEFUL_OUT_BEATS]).copy()
    conv1_total_ms = (time.perf_counter() - stage_t0) * 1000.0

    # ========================================================================
    # POOL1
    # ========================================================================

    stage_t0 = time.perf_counter()
    pool1 = maxpool2x2(conv1_out)
    pool1_ms = (time.perf_counter() - stage_t0) * 1000.0

    # ========================================================================
    # CONV2 -- 16 PASSES
    # ========================================================================

    conv2_stage_t0 = time.perf_counter()

    # Prepare the six padded Pool1 input planes ONCE.
    in_buf[:] = 0

    for ch in range(6):
        plane = np.zeros((256, 256), dtype=np.uint8)
        plane[:13, :13] = pool1[ch]
        in_buf[:, ch] = plane.reshape(-1)

    conv2_out = np.zeros((16, 11, 11), dtype=np.uint8)
    conv2_dma_times = []
    conv2_pass_total_times = []
    conv2_tail_beats = []

    for f in range(16):
        pass_t0 = time.perf_counter()

        out_buf[:] = 0

        soft_reset()
        write_config(shift2, MODE_ACCUMULATE)

        engine.write(REG_ACC_BIAS, u32(conv2_b[f]))

        # Conv2 bias is added once by accumulator, not in individual VEs.
        for e in range(6):
            engine.write(REG_VE_BIAS0 + 4 * e, 0)

        words = conv2_words[f]

        for r in range(14):
            engine.write(REG_WEIGHTS0 + 4 * r, int(words[r]))

        dma_ms, packet_beats = launch_dma(in_buf, out_buf)
        conv2_dma_times.append(dma_ms)

        full_map = np.asarray(
            out_buf[:USEFUL_OUT_BEATS, 0]
        ).reshape(254, 254)

        conv2_out[f] = full_map[:11, :11]
        conv2_tail_beats.append(
            np.asarray(out_buf[USEFUL_OUT_BEATS]).copy()
        )

        conv2_pass_total_times.append(
            (time.perf_counter() - pass_t0) * 1000.0
        )

    conv2_total_ms = (time.perf_counter() - conv2_stage_t0) * 1000.0
    conv2_dma_sum_ms = float(np.sum(conv2_dma_times))

    # ========================================================================
    # POOL2
    # ========================================================================

    stage_t0 = time.perf_counter()
    pool2_hw = maxpool2x2(conv2_out)
    pool2_ms = (time.perf_counter() - stage_t0) * 1000.0

    # ========================================================================
    # FC CLASSIFIER
    # ========================================================================

    stage_t0 = time.perf_counter()

    hw_logits, hw_probs, hw_pred, hw_conf = fc_forward(
        pool2_hw,
        fc1_w, fc1_b,
        fc2_w, fc2_b,
        fc3_w, fc3_b
    )

    fc_ms = (time.perf_counter() - stage_t0) * 1000.0

    inference_ms = (time.perf_counter() - inference_t0) * 1000.0


    # =========================================================================
    # SOFTWARE QUANTIZED REFERENCE
    # =========================================================================

    ref_t0 = time.perf_counter()

    sw_conv1 = conv2d_int_golden(
        test_img.reshape(1, 28, 28),
        conv1_w, conv1_b, shift1
    )

    sw_pool1 = maxpool2x2(sw_conv1)

    sw_conv2 = conv2d_int_golden(
        sw_pool1,
        conv2_w, conv2_b, shift2
    )

    sw_pool2 = maxpool2x2(sw_conv2)

    sw_logits, sw_probs, sw_pred, sw_conf = fc_forward(
        sw_pool2,
        fc1_w, fc1_b,
        fc2_w, fc2_b,
        fc3_w, fc3_b
    )

    reference_ms = (time.perf_counter() - ref_t0) * 1000.0


    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================

    conv1_max_diff = int(
        np.abs(
            conv1_out.astype(np.int16) -
            sw_conv1.astype(np.int16)
        ).max()
    )

    conv1_mismatches = int(
        np.count_nonzero(conv1_out != sw_conv1)
    )

    conv2_abs_diff = np.abs(
        conv2_out.astype(np.int16) -
        sw_conv2.astype(np.int16)
    )

    conv2_max_diff = int(conv2_abs_diff.max())
    conv2_mismatches = int(np.count_nonzero(conv2_out != sw_conv2))

    pool2_abs_diff = np.abs(
        pool2_hw.astype(np.int16) -
        sw_pool2.astype(np.int16)
    )

    pool2_max_diff = int(pool2_abs_diff.max())
    pool2_mismatches = int(np.count_nonzero(pool2_hw != sw_pool2))


    # =========================================================================
    # RESULTS
    # =========================================================================

    print("\n" + "=" * 80)
    print("DIGIT RECOGNITION RESULT")
    print("=" * 80)

    print("Hardware-assisted prediction : %d" % hw_pred)
    print("Hardware confidence          : %.2f%%" % hw_conf)
    print("Hardware probabilities (%)   :",
          np.round(hw_probs * 100.0, 2))

    print("")
    print("Software-reference prediction: %d" % sw_pred)
    print("Software confidence          : %.2f%%" % sw_conf)
    print("Software probabilities (%)   :",
          np.round(sw_probs * 100.0, 2))

    print("")

    if hw_pred == sw_pred:
        print("CLASSIFICATION AGREEMENT      : YES")
    else:
        print("CLASSIFICATION AGREEMENT      : NO")

    print("\n" + "=" * 80)
    print("HARDWARE NUMERICAL CHECK")
    print("=" * 80)

    print("Conv1 max diff / mismatches : %d / %d"
          % (conv1_max_diff, conv1_mismatches))
    print("Conv2 max diff / mismatches : %d / %d"
          % (conv2_max_diff, conv2_mismatches))
    print("Pool2 max diff / mismatches : %d / %d"
          % (pool2_max_diff, pool2_mismatches))

    print("Conv1 tail beat             :", conv1_tail.tolist())

    nonzero_tail_count = 0
    for x in conv2_tail_beats:
        if np.any(x != 0):
            nonzero_tail_count += 1

    print("Conv2 non-zero tail beats   : %d / 16" % nonzero_tail_count)

    print("\n" + "=" * 80)
    print("LATENCY BREAKDOWN")
    print("=" * 80)

    print("Preprocessing               : %9.3f ms" % preprocess_ms)
    print("")
    print("Conv1 DMA + PL              : %9.3f ms" % conv1_dma_ms)
    print("Conv1 total incl. setup     : %9.3f ms" % conv1_total_ms)
    print("Pool1 ARM                   : %9.3f ms" % pool1_ms)
    print("")
    print("Conv2 DMA + PL sum          : %9.3f ms" % conv2_dma_sum_ms)
    print("Conv2 avg DMA/pass          : %9.3f ms"
          % (conv2_dma_sum_ms / 16.0))
    print("Conv2 total incl. setup     : %9.3f ms" % conv2_total_ms)
    print("Conv2 avg total/pass        : %9.3f ms"
          % (float(np.mean(conv2_pass_total_times))))
    print("")
    print("Pool2 ARM                   : %9.3f ms" % pool2_ms)
    print("FC1+FC2+FC3 ARM             : %9.3f ms" % fc_ms)
    print("")
    print("PL/DMA compute-path sum     : %9.3f ms"
          % (conv1_dma_ms + conv2_dma_sum_ms))
    print("Inference, image ready->pred: %9.3f ms" % inference_ms)
    print("Preprocess + inference      : %9.3f ms"
          % (preprocess_ms + inference_ms))

    if inference_ms > 0:
        print("Steady-state throughput     : %9.2f FPS"
              % (1000.0 / inference_ms))

    print("")
    print("SW golden reference time    : %9.3f ms" % reference_ms)

    print("=" * 80)


    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    if SHOW_PLOTS:
        # CNN classification does not reconstruct an output image.
        # We instead visualize the input, internal activation summaries,
        # and final class probabilities.

        plt.figure(figsize=(4, 4))
        plt.imshow(test_img, cmap="gray")
        plt.title("Preprocessed 28x28 input")
        plt.axis("off")
        plt.show()

        # Collapse channels with max() only for visualization.
        pool1_view = np.max(pool1, axis=0)
        plt.figure(figsize=(4, 4))
        plt.imshow(pool1_view, cmap="gray")
        plt.title("Pool1 activation summary (max over 6 channels)")
        plt.axis("off")
        plt.show()

        pool2_view = np.max(pool2_hw, axis=0)
        plt.figure(figsize=(4, 4))
        plt.imshow(pool2_view, cmap="gray")
        plt.title("Pool2 HW activation summary (max over 16 channels)")
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.bar(np.arange(10), hw_probs * 100.0)
        plt.xticks(np.arange(10))
        plt.xlabel("Digit")
        plt.ylabel("Probability (%)")
        plt.title("HW-assisted prediction = %d (%.2f%%)"
                  % (hw_pred, hw_conf))
        plt.show()

finally:
    try:
        in_buf.freebuffer()
    except Exception:
        pass

    try:
        out_buf.freebuffer()
    except Exception:
        pass
