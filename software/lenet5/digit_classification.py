"""
PYNQ-Z2 variable-width CNN accelerator test
=============================================

New hardware:
    Accelerator IP : test_IP_0
    Accelerator    : 0x40000000
    DMA            : axi_dma_0
    DMA base       : 0x40400000

Variable-width operation:
    Conv1:
        input  = 28x28  = 784 beats
        output = 26x26  = 676 beats
        active_width = 28

    Conv2:
        input  = 13x13  = 169 beats
        output = 11x11  = 121 beats
        active_width = 13

NO 256x256 padding.
NO +1 output-beat workaround.

Python 3.6 compatible.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from pynq import Overlay, allocate

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# =============================================================================
# USER SETTINGS
# =============================================================================

BITSTREAM = "test_varwidth.bit"

# Can also use test_image.npy
INPUT_PATH = "digit.png"
FALLBACK_NPY = "test_image.npy"

AUTO_INVERT = True
SHOW_PLOTS = True

# Useful while debugging
DMA_TIMEOUT_S = 2.0
POLL_S = 0.0002


# =============================================================================
# HARDWARE CONSTANTS
# =============================================================================

ACCEL_IP_NAME = "test_IP_0"
DMA_IP_NAME   = "axi_dma_0"

ACCEL_BASE_EXPECTED  = 0x40000000
ACCEL_RANGE_EXPECTED = 0x00001000

DMA_BASE_EXPECTED    = 0x40400000
DMA_RANGE_EXPECTED   = 0x00010000


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------

MODE_ACCUMULATE        = 0
MODE_BYPASS_RAW        = 1
MODE_BYPASS_QUANT      = 2
MODE_BYPASS_RELU_QUANT = 3


# -----------------------------------------------------------------------------
# Register offsets
# -----------------------------------------------------------------------------

REG_CONTROL  = 0x00
REG_STATUS   = 0x04
REG_CONFIG   = 0x08
REG_ACC_BIAS = 0x0C

# VE0 bias = 0x10
# VE1 bias = 0x14
# ...
REG_VE_BIAS0 = 0x10

# Weight registers begin at slv_reg10 = 10*4 = 0x28
REG_WEIGHTS0 = 0x28


# -----------------------------------------------------------------------------
# Actual dimensions
# -----------------------------------------------------------------------------

CONV1_IN_W      = 28
CONV1_OUT_W     = 26
CONV1_IN_BEATS  = 28 * 28
CONV1_OUT_BEATS = 26 * 26

CONV2_IN_W      = 13
CONV2_OUT_W     = 11
CONV2_IN_BEATS  = 13 * 13
CONV2_OUT_BEATS = 11 * 11


def u32(x):
    return int(x) & 0xFFFFFFFF


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def resize_2d_uint8(arr, out_hw):
    if not PIL_AVAILABLE:
        raise RuntimeError(
            "Pillow is required for PNG/JPG preprocessing. "
            "Use test_image.npy instead."
        )

    im = Image.fromarray(arr.astype(np.uint8), mode="L")
    im = im.resize((out_hw[1], out_hw[0]), Image.LANCZOS)

    return np.asarray(im, dtype=np.uint8)


def mnist_like_preprocess(path):
    """
    Returns a 28x28 uint8 image.
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
            "Pillow unavailable. Use test_image.npy."
        )

    im = Image.open(path).convert("L")
    x = np.asarray(im, dtype=np.uint8)

    # Convert black digit / white paper to MNIST polarity.
    if AUTO_INVERT and float(np.mean(x)) > 127.0:
        x = 255 - x

    threshold = max(20, int(0.12 * int(x.max())))
    mask = x > threshold

    if not np.any(mask):
        raise RuntimeError(
            "Could not find visible digit in %s" % path
        )

    rows, cols = np.where(mask)

    r0 = int(rows.min())
    r1 = int(rows.max()) + 1
    c0 = int(cols.min())
    c1 = int(cols.max()) + 1

    crop = x[r0:r1, c0:c1]

    h, w = crop.shape

    scale = min(
        20.0 / float(h),
        20.0 / float(w)
    )

    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    crop = resize_2d_uint8(
        crop,
        (nh, nw)
    )

    out = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    top = (28 - nh) // 2
    left = (28 - nw) // 2

    out[
        top:top + nh,
        left:left + nw
    ] = crop

    return out


# =============================================================================
# MODEL / MATH HELPERS
# =============================================================================

def pack_6x3x3_to_regs(weights_6x3x3):
    """
    6 filters/channels * 3 * 3 * 8-bit = 432 bits.
    AXI register bank provides 14 * 32 = 448 bits.

    Add two padding bytes.
    """

    w = np.asarray(
        weights_6x3x3,
        dtype=np.int8
    ).reshape(6, 3, 3)

    raw = w.tobytes() + b"\x00\x00"

    return np.frombuffer(
        raw,
        dtype=np.uint32
    )


def maxpool2x2(x):

    C, H, W = x.shape

    H2 = (H // 2) * 2
    W2 = (W // 2) * 2

    x = x[:, :H2, :W2]

    return x.reshape(
        C,
        H2 // 2, 2,
        W2 // 2, 2
    ).max(axis=(2, 4)).astype(np.uint8)


def softmax(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    e = np.exp(x - np.max(x))

    return e / np.sum(e)


def conv2d_int_golden(
    pixels,
    weights,
    bias,
    shift
):
    """
    Software integer reference matching accelerator math.
    """

    Cout, Cin, K, _ = weights.shape

    H = pixels.shape[1]
    W = pixels.shape[2]

    Ho = H - K + 1
    Wo = W - K + 1

    out = np.zeros(
        (Cout, Ho, Wo),
        dtype=np.int64
    )

    for co in range(Cout):

        acc = np.zeros(
            (Ho, Wo),
            dtype=np.int64
        )

        for ci in range(Cin):

            for ky in range(K):

                for kx in range(K):

                    acc += (
                        pixels[
                            ci,
                            ky:ky + Ho,
                            kx:kx + Wo
                        ].astype(np.int64)
                        *
                        int(
                            weights[
                                co,
                                ci,
                                ky,
                                kx
                            ]
                        )
                    )

        acc += int(bias[co])

        acc = np.where(
            acc < 0,
            0,
            acc
        )

        acc = acc >> int(shift)

        out[co] = np.clip(
            acc,
            0,
            255
        )

    return out.astype(np.uint8)


def fc_forward(
    pool2,
    fc1_w,
    fc1_b,
    fc2_w,
    fc2_b,
    fc3_w,
    fc3_b
):

    feat = pool2.astype(
        np.float32
    ).reshape(400)

    x1 = np.dot(
        fc1_w,
        feat
    ) + fc1_b

    x1 = np.maximum(
        x1,
        0
    )

    x2 = np.dot(
        fc2_w,
        x1
    ) + fc2_b

    x2 = np.maximum(
        x2,
        0
    )

    logits = np.dot(
        fc3_w,
        x2
    ) + fc3_b

    probs = softmax(logits)

    pred = int(
        np.argmax(probs)
    )

    confidence = float(
        probs[pred] * 100.0
    )

    return (
        logits,
        probs,
        pred,
        confidence
    )


# =============================================================================
# LOAD OVERLAY
# =============================================================================

if not os.path.exists(BITSTREAM):
    raise FileNotFoundError(
        "Could not find %s" % BITSTREAM
    )


print("=" * 80)
print("VARIABLE-WIDTH FPGA CNN TEST")
print("=" * 80)

print("Loading overlay:", BITSTREAM)

overlay = Overlay(BITSTREAM)


print(
    "Available IPs:",
    sorted(overlay.ip_dict.keys())
)


if ACCEL_IP_NAME not in overlay.ip_dict:

    raise RuntimeError(
        "Expected accelerator '%s' not found.\nAvailable IPs: %s"
        % (
            ACCEL_IP_NAME,
            sorted(overlay.ip_dict.keys())
        )
    )


if DMA_IP_NAME not in overlay.ip_dict:

    raise RuntimeError(
        "Expected DMA '%s' not found.\nAvailable IPs: %s"
        % (
            DMA_IP_NAME,
            sorted(overlay.ip_dict.keys())
        )
    )


acc_info = overlay.ip_dict[
    ACCEL_IP_NAME
]

dma_info = overlay.ip_dict[
    DMA_IP_NAME
]


acc_base = int(
    acc_info.get(
        "phys_addr",
        -1
    )
)

acc_range = int(
    acc_info.get(
        "addr_range",
        -1
    )
)

dma_base = int(
    dma_info.get(
        "phys_addr",
        -1
    )
)

dma_range = int(
    dma_info.get(
        "addr_range",
        -1
    )
)


print(
    "Accelerator: base=0x%08X range=0x%X"
    % (
        acc_base,
        acc_range
    )
)

print(
    "DMA        : base=0x%08X range=0x%X"
    % (
        dma_base,
        dma_range
    )
)


if acc_base != ACCEL_BASE_EXPECTED:

    raise RuntimeError(
        "Accelerator base address mismatch."
    )


if dma_base != DMA_BASE_EXPECTED:

    raise RuntimeError(
        "DMA base address mismatch."
    )


engine = getattr(
    overlay,
    ACCEL_IP_NAME
)

dma = getattr(
    overlay,
    DMA_IP_NAME
)


print("Address map: VERIFIED")


# =============================================================================
# LOAD MODEL FILES
# =============================================================================

conv1_w = np.load(
    "conv1_weights_int8.npy"
).astype(np.int8)

conv1_b = np.load(
    "conv1_bias_int32.npy"
).astype(np.int32)


conv2_w = np.load(
    "conv2_weights_int8.npy"
).astype(np.int8)

conv2_b = np.load(
    "conv2_bias_int32.npy"
).astype(np.int32)


shift1, shift2 = [
    int(x)
    for x in np.load(
        "shift_amounts.npy"
    )
]


fc1_w = np.load(
    "fc1_weights.npy"
).astype(np.float32)

fc1_b = np.load(
    "fc1_bias.npy"
).astype(np.float32)


fc2_w = np.load(
    "fc2_weights.npy"
).astype(np.float32)

fc2_b = np.load(
    "fc2_bias.npy"
).astype(np.float32)


fc3_w = np.load(
    "fc3_weights.npy"
).astype(np.float32)

fc3_b = np.load(
    "fc3_bias.npy"
).astype(np.float32)


# -----------------------------------------------------------------------------
# Shape checks
# -----------------------------------------------------------------------------

assert conv1_w.shape == (
    6,
    1,
    3,
    3
)

assert conv2_w.shape == (
    16,
    6,
    3,
    3
)

assert fc1_w.shape == (
    120,
    400
)

assert fc2_w.shape == (
    84,
    120
)

assert fc3_w.shape == (
    10,
    84
)


conv1_words = pack_6x3x3_to_regs(
    conv1_w[:, 0]
)

conv2_words = [
    pack_6x3x3_to_regs(
        conv2_w[f]
    )
    for f in range(16)
]


print("Model files: LOADED")
print(
    "shift1=%d shift2=%d"
    % (
        shift1,
        shift2
    )
)


# =============================================================================
# INPUT IMAGE
# =============================================================================

input_path = INPUT_PATH


if not os.path.exists(input_path):

    if os.path.exists(FALLBACK_NPY):

        input_path = FALLBACK_NPY

        print(
            "Input '%s' not found; using '%s'."
            % (
                INPUT_PATH,
                input_path
            )
        )

    else:

        raise FileNotFoundError(
            "Put '%s' or '%s' in the Jupyter folder."
            % (
                INPUT_PATH,
                FALLBACK_NPY
            )
        )


t0 = time.perf_counter()

test_img = mnist_like_preprocess(
    input_path
)

preprocess_ms = (
    time.perf_counter() - t0
) * 1000.0


print("")
print("Input:", input_path)

print(
    "Preprocessed image:",
    test_img.shape,
    test_img.dtype,
    "min=%d max=%d"
    % (
        int(test_img.min()),
        int(test_img.max())
    )
)

print(
    "Preprocess time: %.3f ms"
    % preprocess_ms
)


# =============================================================================
# HARDWARE HELPERS
# =============================================================================

def soft_reset():

    engine.write(
        REG_CONTROL,
        0x02
    )

    time.sleep(
        0.0002
    )

    engine.write(
        REG_CONTROL,
        0x00
    )


def write_config(
    shift,
    mode,
    active_width,
    expected_beats
):
    """
    REG_CONFIG / slv_reg2

    [31:16] expected_beats
    [15:7]  active_width
    [6:2]   shift
    [1:0]   mode
    """

    cfg = (
        (
            int(expected_beats)
            & 0xFFFF
        ) << 16
    )

    cfg |= (
        (
            int(active_width)
            & 0x1FF
        ) << 7
    )

    cfg |= (
        (
            int(shift)
            & 0x1F
        ) << 2
    )

    cfg |= (
        int(mode)
        & 0x3
    )

    engine.write(
        REG_CONFIG,
        u32(cfg)
    )


def read_dmasr(channel):

    return int(
        channel._mmio.read(
            channel._offset + 4
        )
    )


def read_dma_length(channel):

    return int(
        channel._mmio.read(
            channel._offset + 0x28
        )
    )


def launch_dma(
    in_buf,
    out_buf,
    expected_beats
):
    """
    One accelerator transaction.

    Sequence:
      1. arm receive DMA
      2. assert accelerator master_start
      3. check ready
      4. start transmit DMA
      5. wait for both DMA channels
    """

    # -------------------------------------------------------------------------
    # Arm S2MM before anything can generate outputs
    # -------------------------------------------------------------------------

    dma.recvchannel.transfer(
        out_buf
    )


    # -------------------------------------------------------------------------
    # Start accelerator
    # -------------------------------------------------------------------------

    engine.write(
        REG_CONTROL,
        0x01
    )


    status = int(
        engine.read(
            REG_STATUS
        )
    )

    ready = (
        status >> 1
    ) & 1


    if not ready:

        engine.write(
            REG_CONTROL,
            0x00
        )

        raise RuntimeError(
            "Accelerator not ready after master_start. "
            "STATUS=0x%08X"
            % status
        )


    # -------------------------------------------------------------------------
    # Start timing immediately before MM2S
    # -------------------------------------------------------------------------

    t0 = time.perf_counter()


    dma.sendchannel.transfer(
        in_buf
    )


    while True:

        send_stat = read_dmasr(
            dma.sendchannel
        )

        recv_stat = read_dmasr(
            dma.recvchannel
        )


        # DMA error bits
        if send_stat & 0x70:

            engine.write(
                REG_CONTROL,
                0x00
            )

            raise RuntimeError(
                "MM2S DMA error: 0x%08X"
                % send_stat
            )


        if recv_stat & 0x70:

            engine.write(
                REG_CONTROL,
                0x00
            )

            raise RuntimeError(
                "S2MM DMA error: 0x%08X"
                % recv_stat
            )


        # Both idle
        if (
            (send_stat & 0x2)
            and
            (recv_stat & 0x2)
        ):
            break


        if (
            time.perf_counter() - t0
        ) > DMA_TIMEOUT_S:

            engine.write(
                REG_CONTROL,
                0x00
            )

            raise RuntimeError(
                "DMA timeout. "
                "MM2S=0x%08X "
                "S2MM=0x%08X"
                % (
                    send_stat,
                    recv_stat
                )
            )


        time.sleep(
            POLL_S
        )


    t1 = time.perf_counter()


    dma.sendchannel.wait()
    dma.recvchannel.wait()


    try:
        out_buf.invalidate()
    except Exception:
        pass


    received_bytes = read_dma_length(
        dma.recvchannel
    )

    received_beats = (
        received_bytes // 8
    )


    engine.write(
        REG_CONTROL,
        0x00
    )


    if (
        received_beats
        != expected_beats
    ):

        raise RuntimeError(
            "Received %d beats; expected exactly %d."
            % (
                received_beats,
                expected_beats
            )
        )


    return (
        (t1 - t0) * 1000.0,
        received_beats
    )


# =============================================================================
# HARDWARE INFERENCE
# =============================================================================

conv1_in = None
conv1_out_buf = None

conv2_in = None
conv2_out_buf = None


try:

    print("")
    print("=" * 80)
    print("HARDWARE-ASSISTED INFERENCE")
    print("=" * 80)


    inference_t0 = time.perf_counter()


    # =========================================================================
    # CONV1
    # =========================================================================

    print("")
    print("CONV1")
    print("-" * 80)


    stage_t0 = time.perf_counter()


    conv1_in = allocate(
        shape=(
            CONV1_IN_BEATS,
            8
        ),
        dtype=np.uint8
    )


    conv1_out_buf = allocate(
        shape=(
            CONV1_OUT_BEATS,
            8
        ),
        dtype=np.uint8
    )


    # Actual 28x28 input.
    # No 256x256 padding.
    flat = test_img.reshape(-1)


    conv1_in[:] = 0


    # Same image to all six engines.
    for ch in range(6):

        conv1_in[:, ch] = flat


    conv1_out_buf[:] = 0


    try:
        conv1_in.flush()
    except Exception:
        pass


    soft_reset()


    write_config(
        shift=shift1,
        mode=MODE_BYPASS_RELU_QUANT,
        active_width=CONV1_IN_W,
        expected_beats=CONV1_OUT_BEATS
    )


    # Accumulator bias unused in bypass mode.
    engine.write(
        REG_ACC_BIAS,
        0
    )


    # Six Conv1 filters have individual biases.
    for e in range(6):

        engine.write(
            REG_VE_BIAS0 + 4 * e,
            u32(conv1_b[e])
        )


    # 14 32-bit registers = 56 bytes
    for r in range(14):

        engine.write(
            REG_WEIGHTS0 + 4 * r,
            int(conv1_words[r])
        )


    conv1_dma_ms, conv1_received = launch_dma(
        conv1_in,
        conv1_out_buf,
        CONV1_OUT_BEATS
    )


    print(
        "Input beats   : %d"
        % CONV1_IN_BEATS
    )

    print(
        "Output beats  : %d"
        % conv1_received
    )

    print(
        "Active width  : %d"
        % CONV1_IN_W
    )

    print(
        "DMA + PL time : %.3f ms"
        % conv1_dma_ms
    )


    # -------------------------------------------------------------------------
    # Extract all six output channels directly.
    # Output is already exactly 26x26.
    # -------------------------------------------------------------------------

    conv1_out = np.zeros(
        (
            6,
            CONV1_OUT_W,
            CONV1_OUT_W
        ),
        dtype=np.uint8
    )


    for ch in range(6):

        conv1_out[ch] = np.asarray(
            conv1_out_buf[:, ch]
        ).reshape(
            CONV1_OUT_W,
            CONV1_OUT_W
        )


    conv1_total_ms = (
        time.perf_counter()
        - stage_t0
    ) * 1000.0


    # Free Conv1 DMA buffers now.
    conv1_in.freebuffer()
    conv1_out_buf.freebuffer()

    conv1_in = None
    conv1_out_buf = None


    # =========================================================================
    # POOL1
    # =========================================================================

    stage_t0 = time.perf_counter()


    pool1 = maxpool2x2(
        conv1_out
    )


    pool1_ms = (
        time.perf_counter()
        - stage_t0
    ) * 1000.0


    print("")
    print(
        "Pool1 shape   : %s"
        % (
            str(pool1.shape)
        )
    )


    # =========================================================================
    # CONV2
    # =========================================================================

    print("")
    print("CONV2")
    print("-" * 80)


    conv2_stage_t0 = time.perf_counter()


    conv2_in = allocate(
        shape=(
            CONV2_IN_BEATS,
            8
        ),
        dtype=np.uint8
    )


    conv2_out_buf = allocate(
        shape=(
            CONV2_OUT_BEATS,
            8
        ),
        dtype=np.uint8
    )


    # -------------------------------------------------------------------------
    # Pack six real 13x13 Pool1 maps directly.
    # No 256x256 padding.
    # -------------------------------------------------------------------------

    conv2_in[:] = 0


    for ch in range(6):

        conv2_in[:, ch] = (
            pool1[ch]
            .reshape(-1)
        )


    try:
        conv2_in.flush()
    except Exception:
        pass


    conv2_out = np.zeros(
        (
            16,
            CONV2_OUT_W,
            CONV2_OUT_W
        ),
        dtype=np.uint8
    )


    conv2_dma_times = []
    conv2_pass_total_times = []


    for f in range(16):

        pass_t0 = time.perf_counter()


        conv2_out_buf[:] = 0


        soft_reset()


        write_config(
            shift=shift2,
            mode=MODE_ACCUMULATE,
            active_width=CONV2_IN_W,
            expected_beats=CONV2_OUT_BEATS
        )


        # Conv2 final filter bias is added once
        # after six-channel accumulation.
        engine.write(
            REG_ACC_BIAS,
            u32(conv2_b[f])
        )


        # Individual VEs must NOT add bias in Conv2.
        for e in range(6):

            engine.write(
                REG_VE_BIAS0 + 4 * e,
                0
            )


        words = conv2_words[f]


        for r in range(14):

            engine.write(
                REG_WEIGHTS0 + 4 * r,
                int(words[r])
            )


        dma_ms, received = launch_dma(
            conv2_in,
            conv2_out_buf,
            CONV2_OUT_BEATS
        )


        conv2_dma_times.append(
            dma_ms
        )


        if received != CONV2_OUT_BEATS:

            raise RuntimeError(
                "Conv2 filter %d produced %d beats; expected %d."
                % (
                    f,
                    received,
                    CONV2_OUT_BEATS
                )
            )


        # Accumulated output is in byte lane 0.
        conv2_out[f] = np.asarray(
            conv2_out_buf[:, 0]
        ).reshape(
            CONV2_OUT_W,
            CONV2_OUT_W
        )


        pass_ms = (
            time.perf_counter()
            - pass_t0
        ) * 1000.0


        conv2_pass_total_times.append(
            pass_ms
        )


        print(
            "Filter %2d : %3d beats | DMA+PL %.3f ms | total %.3f ms"
            % (
                f,
                received,
                dma_ms,
                pass_ms
            )
        )


    conv2_total_ms = (
        time.perf_counter()
        - conv2_stage_t0
    ) * 1000.0


    conv2_dma_sum_ms = float(
        np.sum(
            conv2_dma_times
        )
    )


    conv2_in.freebuffer()
    conv2_out_buf.freebuffer()

    conv2_in = None
    conv2_out_buf = None


    # =========================================================================
    # POOL2
    # =========================================================================

    stage_t0 = time.perf_counter()


    pool2_hw = maxpool2x2(
        conv2_out
    )


    pool2_ms = (
        time.perf_counter()
        - stage_t0
    ) * 1000.0


    # =========================================================================
    # FC CLASSIFIER
    # =========================================================================

    stage_t0 = time.perf_counter()


    (
        hw_logits,
        hw_probs,
        hw_pred,
        hw_conf
    ) = fc_forward(
        pool2_hw,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b
    )


    fc_ms = (
        time.perf_counter()
        - stage_t0
    ) * 1000.0


    inference_ms = (
        time.perf_counter()
        - inference_t0
    ) * 1000.0


    # =========================================================================
    # SOFTWARE GOLDEN REFERENCE
    # =========================================================================

    ref_t0 = time.perf_counter()


    sw_conv1 = conv2d_int_golden(
        test_img.reshape(
            1,
            28,
            28
        ),
        conv1_w,
        conv1_b,
        shift1
    )


    sw_pool1 = maxpool2x2(
        sw_conv1
    )


    sw_conv2 = conv2d_int_golden(
        sw_pool1,
        conv2_w,
        conv2_b,
        shift2
    )


    sw_pool2 = maxpool2x2(
        sw_conv2
    )


    (
        sw_logits,
        sw_probs,
        sw_pred,
        sw_conf
    ) = fc_forward(
        sw_pool2,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b
    )


    reference_ms = (
        time.perf_counter()
        - ref_t0
    ) * 1000.0


    # =========================================================================
    # NUMERICAL CHECK
    # =========================================================================

    conv1_diff = np.abs(
        conv1_out.astype(np.int16)
        -
        sw_conv1.astype(np.int16)
    )


    conv1_max_diff = int(
        conv1_diff.max()
    )

    conv1_mismatches = int(
        np.count_nonzero(
            conv1_out
            !=
            sw_conv1
        )
    )


    conv2_diff = np.abs(
        conv2_out.astype(np.int16)
        -
        sw_conv2.astype(np.int16)
    )


    conv2_max_diff = int(
        conv2_diff.max()
    )

    conv2_mismatches = int(
        np.count_nonzero(
            conv2_out
            !=
            sw_conv2
        )
    )


    pool2_diff = np.abs(
        pool2_hw.astype(np.int16)
        -
        sw_pool2.astype(np.int16)
    )


    pool2_max_diff = int(
        pool2_diff.max()
    )

    pool2_mismatches = int(
        np.count_nonzero(
            pool2_hw
            !=
            sw_pool2
        )
    )


    # =========================================================================
    # RESULTS
    # =========================================================================

    print("")
    print("=" * 80)
    print("VARIABLE-WIDTH HARDWARE CHECK")
    print("=" * 80)


    print(
        "Conv1 input/output beats : %d / %d"
        % (
            CONV1_IN_BEATS,
            CONV1_OUT_BEATS
        )
    )

    print(
        "Conv2 input/output beats : %d / %d"
        % (
            CONV2_IN_BEATS,
            CONV2_OUT_BEATS
        )
    )


    print("")
    print(
        "Conv1 max diff / mismatches : %d / %d"
        % (
            conv1_max_diff,
            conv1_mismatches
        )
    )

    print(
        "Conv2 max diff / mismatches : %d / %d"
        % (
            conv2_max_diff,
            conv2_mismatches
        )
    )

    print(
        "Pool2 max diff / mismatches : %d / %d"
        % (
            pool2_max_diff,
            pool2_mismatches
        )
    )


    print("")
    print("=" * 80)
    print("DIGIT RECOGNITION")
    print("=" * 80)


    print(
        "Hardware prediction : %d"
        % hw_pred
    )

    print(
        "Hardware confidence : %.2f%%"
        % hw_conf
    )

    print(
        "Hardware probabilities:",
        np.round(
            hw_probs * 100.0,
            2
        )
    )


    print("")


    print(
        "Software prediction : %d"
        % sw_pred
    )

    print(
        "Software confidence : %.2f%%"
        % sw_conf
    )

    print(
        "Software probabilities:",
        np.round(
            sw_probs * 100.0,
            2
        )
    )


    print("")


    if hw_pred == sw_pred:
        print(
            "CLASSIFICATION AGREEMENT: YES"
        )
    else:
        print(
            "CLASSIFICATION AGREEMENT: NO"
        )


    print("")
    print("=" * 80)
    print("LATENCY")
    print("=" * 80)


    print(
        "Preprocessing               : %9.3f ms"
        % preprocess_ms
    )

    print("")

    print(
        "Conv1 DMA + PL              : %9.3f ms"
        % conv1_dma_ms
    )

    print(
        "Conv1 total incl. setup     : %9.3f ms"
        % conv1_total_ms
    )

    print(
        "Pool1 ARM                   : %9.3f ms"
        % pool1_ms
    )

    print("")

    print(
        "Conv2 DMA + PL sum          : %9.3f ms"
        % conv2_dma_sum_ms
    )

    print(
        "Conv2 average DMA/pass      : %9.3f ms"
        % (
            conv2_dma_sum_ms
            / 16.0
        )
    )

    print(
        "Conv2 total incl. setup     : %9.3f ms"
        % conv2_total_ms
    )

    print(
        "Conv2 average total/pass    : %9.3f ms"
        % float(
            np.mean(
                conv2_pass_total_times
            )
        )
    )

    print("")

    print(
        "Pool2 ARM                   : %9.3f ms"
        % pool2_ms
    )

    print(
        "FC1+FC2+FC3 ARM             : %9.3f ms"
        % fc_ms
    )

    print("")

    print(
        "PL/DMA convolution sum      : %9.3f ms"
        % (
            conv1_dma_ms
            +
            conv2_dma_sum_ms
        )
    )

    print(
        "Inference image-ready->pred : %9.3f ms"
        % inference_ms
    )

    print(
        "Preprocess + inference      : %9.3f ms"
        % (
            preprocess_ms
            +
            inference_ms
        )
    )


    if inference_ms > 0:

        print(
            "Steady-state throughput     : %9.2f FPS"
            % (
                1000.0
                / inference_ms
            )
        )


    print("")

    print(
        "SW golden reference time    : %9.3f ms"
        % reference_ms
    )


    print("=" * 80)


    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    if SHOW_PLOTS:

        plt.figure(
            figsize=(4, 4)
        )

        plt.imshow(
            test_img,
            cmap="gray"
        )

        plt.title(
            "Preprocessed 28x28 input"
        )

        plt.axis("off")

        plt.show()


        pool1_view = np.max(
            pool1,
            axis=0
        )

        plt.figure(
            figsize=(4, 4)
        )

        plt.imshow(
            pool1_view,
            cmap="gray"
        )

        plt.title(
            "Pool1 activation summary"
        )

        plt.axis("off")

        plt.show()


        pool2_view = np.max(
            pool2_hw,
            axis=0
        )

        plt.figure(
            figsize=(4, 4)
        )

        plt.imshow(
            pool2_view,
            cmap="gray"
        )

        plt.title(
            "Pool2 HW activation summary"
        )

        plt.axis("off")

        plt.show()


        plt.figure(
            figsize=(7, 4)
        )

        plt.bar(
            np.arange(10),
            hw_probs * 100.0
        )

        plt.xticks(
            np.arange(10)
        )

        plt.xlabel(
            "Digit"
        )

        plt.ylabel(
            "Probability (%)"
        )

        plt.title(
            "HW prediction = %d (%.2f%%)"
            % (
                hw_pred,
                hw_conf
            )
        )

        plt.show()


finally:

    if conv1_in is not None:
        try:
            conv1_in.freebuffer()
        except Exception:
            pass

    if conv1_out_buf is not None:
        try:
            conv1_out_buf.freebuffer()
        except Exception:
            pass

    if conv2_in is not None:
        try:
            conv2_in.freebuffer()
        except Exception:
            pass

    if conv2_out_buf is not None:
        try:
            conv2_out_buf.freebuffer()
        except Exception:
            pass
