# =============================================================================
# PYNQ-Z2 LIVE WEBCAM DIGIT CLASSIFIER - OPTIMIZED VERSION
# =============================================================================
#
# Logitech C270 -> OpenCV -> 28x28 -> FPGA Conv1 -> ARM Pool1
# -> FPGA Conv2 x16 -> ARM Pool2 + FC -> prediction
#
# Optimizations:
#   1. Camera capture in separate thread
#   2. Jupyter display in separate thread
#   3. No clear_output()
#   4. Persistent DMA buffers
#   5. Fast contour-based preprocessing
#   6. Hidden larger capture ROI so digit is not cut
#   7. Batched AXI-Lite parameter writes
#   8. No artificial reset/poll sleeps
#
# Python 3.6 compatible
# =============================================================================

import os
import time
import threading

import cv2
import numpy as np
import ipywidgets as widgets

from pynq import Overlay, allocate
from IPython.display import display


# =============================================================================
# SETTINGS
# =============================================================================

BITSTREAM = "test_varwidth.bit"

ACCEL_IP_NAME = "test_IP_0"
DMA_IP_NAME   = "axi_dma_0"

CAMERA_INDEX = 0

CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

# What you SEE on screen
GUIDE_SIZE = 240

# What is ACTUALLY captured for classification.
# Larger than GUIDE_SIZE so strokes close to the square are not cut.
CAPTURE_SIZE = 320

# Display doesn't need to run as fast as inference.
DISPLAY_INTERVAL = 0.10       # ~10 display updates/sec

JPEG_WIDTH   = 480
JPEG_HEIGHT  = 360
JPEG_QUALITY = 55

DMA_TIMEOUT_S = 2.0


try:
    cv2.setNumThreads(1)
except Exception:
    pass


# =============================================================================
# REGISTER MAP
# =============================================================================

REG_CONTROL  = 0x00
REG_STATUS   = 0x04

REG_CONFIG   = 0x08      # slv_reg2
REG_ACC_BIAS = 0x0C      # slv_reg3
REG_VE_BIAS0 = 0x10      # slv_reg4
REG_WEIGHTS0 = 0x28      # slv_reg10


MODE_ACCUMULATE        = 0
MODE_BYPASS_RAW        = 1
MODE_BYPASS_QUANT      = 2
MODE_BYPASS_RELU_QUANT = 3


# =============================================================================
# DIMENSIONS
# =============================================================================

CONV1_IN_W      = 28
CONV1_OUT_W     = 26

CONV1_IN_BEATS  = 28 * 28
CONV1_OUT_BEATS = 26 * 26


CONV2_IN_W      = 13
CONV2_OUT_W     = 11

CONV2_IN_BEATS  = 13 * 13
CONV2_OUT_BEATS = 11 * 11


# =============================================================================
# HELPERS
# =============================================================================

def u32(x):
    return int(x) & 0xFFFFFFFF


def pack_config(
    shift,
    mode,
    active_width,
    expected_beats
):

    return u32(
        ((int(expected_beats) & 0xFFFF) << 16)
        |
        ((int(active_width) & 0x1FF) << 7)
        |
        ((int(shift) & 0x1F) << 2)
        |
        (int(mode) & 0x3)
    )


def pack_6x3x3_to_regs(weights_6x3x3):
    """
    Preserve exactly the packing convention that already passed your
    bit-exact variable-width test.
    """

    w = np.asarray(
        weights_6x3x3,
        dtype=np.int8
    ).reshape(6, 3, 3)

    raw = w.tobytes() + b"\x00\x00"

    return np.frombuffer(
        raw,
        dtype=np.uint32
    ).copy()


def maxpool2x2(x):

    C, H, W = x.shape

    H2 = (H // 2) * 2
    W2 = (W // 2) * 2

    x = x[:, :H2, :W2]

    return x.reshape(
        C,
        H2 // 2,
        2,
        W2 // 2,
        2
    ).max(
        axis=(2, 4)
    ).astype(np.uint8)


def softmax(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    x = x - np.max(x)

    e = np.exp(x)

    return e / np.sum(e)


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
        pred,
        confidence,
        probs
    )


# =============================================================================
# LOAD OVERLAY
# =============================================================================

if not os.path.exists(BITSTREAM):

    raise FileNotFoundError(
        "Could not find %s"
        % BITSTREAM
    )


print("=" * 80)
print("OPTIMIZED LIVE FPGA DIGIT CLASSIFIER")
print("=" * 80)

print("Loading overlay...")


overlay = Overlay(
    BITSTREAM
)


print(
    "Available IPs:",
    sorted(
        overlay.ip_dict.keys()
    )
)


if ACCEL_IP_NAME not in overlay.ip_dict:

    raise RuntimeError(
        "Accelerator '%s' not found.\nAvailable: %s"
        % (
            ACCEL_IP_NAME,
            sorted(overlay.ip_dict.keys())
        )
    )


if DMA_IP_NAME not in overlay.ip_dict:

    raise RuntimeError(
        "DMA '%s' not found."
        % DMA_IP_NAME
    )


engine = getattr(
    overlay,
    ACCEL_IP_NAME
)

dma = getattr(
    overlay,
    DMA_IP_NAME
)


print("Overlay loaded.")


# =============================================================================
# LOAD MODEL
# =============================================================================

print("Loading model...")


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


assert conv1_w.shape == (6, 1, 3, 3)
assert conv2_w.shape == (16, 6, 3, 3)

assert fc1_w.shape == (120, 400)
assert fc2_w.shape == (84, 120)
assert fc3_w.shape == (10, 84)


conv1_words = pack_6x3x3_to_regs(
    conv1_w[:, 0]
)


conv2_words = [
    pack_6x3x3_to_regs(
        conv2_w[f]
    )
    for f in range(16)
]


print(
    "Model loaded. shift1=%d shift2=%d"
    % (
        shift1,
        shift2
    )
)


# =============================================================================
# PRECOMPUTE REGISTER BLOCKS
# =============================================================================
#
# Registers are contiguous:
#
#   0x08 REG_CONFIG
#   0x0C REG_ACC_BIAS
#   0x10 VE bias 0
#   ...
#   0x24 VE bias 5
#   0x28 weight word 0
#   ...
#   0x5C weight word 13
#
# Therefore we can write:
#
#   2 + 6 + 14 = 22 registers
#
# as ONE NumPy/MMIO slice instead of 22 Python engine.write() calls.
# =============================================================================

CONV1_CONFIG = pack_config(
    shift=shift1,
    mode=MODE_BYPASS_RELU_QUANT,
    active_width=28,
    expected_beats=676
)


CONV2_CONFIG = pack_config(
    shift=shift2,
    mode=MODE_ACCUMULATE,
    active_width=13,
    expected_beats=121
)


conv1_register_block = np.zeros(
    22,
    dtype=np.uint32
)

conv1_register_block[0] = CONV1_CONFIG
conv1_register_block[1] = 0

conv1_register_block[2:8] = np.asarray(
    [u32(x) for x in conv1_b],
    dtype=np.uint32
)

conv1_register_block[8:22] = conv1_words


conv2_register_blocks = []


for f in range(16):

    block = np.zeros(
        22,
        dtype=np.uint32
    )

    block[0] = CONV2_CONFIG

    block[1] = u32(
        conv2_b[f]
    )

    # block[2:8] remain zero:
    # VE biases are zero in Conv2.

    block[8:22] = conv2_words[f]

    conv2_register_blocks.append(
        block
    )


def write_register_block(block):

    start = REG_CONFIG // 4

    engine.mmio.array[
        start:start + 22
    ] = block


# =============================================================================
# HARDWARE CONTROL
# =============================================================================

def soft_reset():
    """
    AXI-Lite writes take many PL clock cycles, so no artificial 100 us sleep
    is required between asserting and clearing the synchronous soft reset.
    """

    engine.write(
        REG_CONTROL,
        0x02
    )

    engine.write(
        REG_CONTROL,
        0x00
    )


def read_dmasr(channel):

    return int(
        channel._mmio.read(
            channel._offset + 4
        )
    )


def launch_dma(
    in_buf,
    out_buf
):
    """
    S2MM first -> accelerator start -> MM2S.

    Uses the exact-sized output buffer, so TLAST terminates the transaction.
    """

    try:
        in_buf.flush()
    except Exception:
        pass


    # Receiver MUST be ready before accelerator can produce output.
    dma.recvchannel.transfer(
        out_buf
    )


    # Assert master_start
    engine.write(
        REG_CONTROL,
        0x01
    )


    t0 = time.perf_counter()


    dma.sendchannel.transfer(
        in_buf
    )


    # PYNQ wait() already performs the required DMA completion polling.
    dma.sendchannel.wait()
    dma.recvchannel.wait()


    t1 = time.perf_counter()


    try:
        out_buf.invalidate()
    except Exception:
        pass


    # Deassert master_start
    engine.write(
        REG_CONTROL,
        0x00
    )


    return (
        t1 - t0
    ) * 1000.0


# =============================================================================
# ALLOCATE DMA BUFFERS ONCE
# =============================================================================

print("Allocating DMA buffers...")


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


print("Buffers allocated.")


# =============================================================================
# OPTIMIZED INFERENCE
# =============================================================================

def infer_digit(
    digit28
):

    inference_t0 = time.perf_counter()


    # =========================================================================
    # CONV1 INPUT
    # =========================================================================

    flat = digit28.reshape(-1)


    conv1_in[:] = 0


    # Replicate same image to six VEs in one NumPy operation.
    conv1_in[:, 0:6] = (
        flat[:, None]
    )


    # =========================================================================
    # CONV1 CONFIG + WEIGHTS
    # =========================================================================

    soft_reset()


    write_register_block(
        conv1_register_block
    )


    conv1_dma_ms = launch_dma(
        conv1_in,
        conv1_out_buf
    )


    # =========================================================================
    # CONV1 OUTPUT
    # =========================================================================

    raw_conv1 = np.asarray(
        conv1_out_buf[:, 0:6]
    )


    # [676,6] -> [6,676] -> [6,26,26]
    conv1_out = raw_conv1.T.reshape(
        6,
        26,
        26
    ).copy()


    # =========================================================================
    # POOL1
    # =========================================================================

    pool1 = maxpool2x2(
        conv1_out
    )


    # =========================================================================
    # PREPARE CONV2 INPUT
    # =========================================================================

    conv2_in[:] = 0


    # pool1:
    #     [6,13,13]
    #
    # reshape:
    #     [6,169]
    #
    # transpose:
    #     [169,6]
    #
    conv2_in[:, 0:6] = (
        pool1
        .reshape(6, -1)
        .T
    )


    conv2_dma_sum_ms = 0.0


    conv2_out = np.zeros(
        (
            16,
            11,
            11
        ),
        dtype=np.uint8
    )


    # =========================================================================
    # CONV2 - 16 FILTER PASSES
    # =========================================================================

    for f in range(16):

        soft_reset()


        # ONE 22-word MMIO block write instead of:
        #   config
        #   accumulator bias
        #   6 VE biases
        #   14 weight writes
        write_register_block(
            conv2_register_blocks[f]
        )


        dma_ms = launch_dma(
            conv2_in,
            conv2_out_buf
        )


        conv2_dma_sum_ms += dma_ms


        conv2_out[f] = np.asarray(
            conv2_out_buf[:, 0]
        ).reshape(
            11,
            11
        )


    # =========================================================================
    # POOL2
    # =========================================================================

    pool2 = maxpool2x2(
        conv2_out
    )


    # =========================================================================
    # FC
    # =========================================================================

    pred, confidence, probs = fc_forward(
        pool2,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b
    )


    inference_ms = (
        time.perf_counter()
        - inference_t0
    ) * 1000.0


    return (
        pred,
        confidence,
        probs,
        inference_ms,
        conv1_dma_ms,
        conv2_dma_sum_ms
    )


# =============================================================================
# CAMERA
# =============================================================================

class CameraThread(object):

    def __init__(
        self,
        index=0
    ):

        self.cap = cv2.VideoCapture(
            index
        )


        if not self.cap.isOpened():

            raise RuntimeError(
                "Could not open Logitech C270 at camera index %d."
                % index
            )


        # Ask C270 for MJPEG.
        # This normally reduces USB traffic.
        try:

            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(
                    'M',
                    'J',
                    'P',
                    'G'
                )
            )

        except Exception:
            pass


        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )


        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )


        self.cap.set(
            cv2.CAP_PROP_FPS,
            30
        )


        try:

            self.cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

        except Exception:
            pass


        self.frame = None

        self.lock = threading.Lock()

        self.running = True


        self.thread = threading.Thread(
            target=self._capture_loop
        )

        self.thread.daemon = True

        self.thread.start()


    def _capture_loop(self):

        while self.running:

            ok, frame = self.cap.read()


            if not ok:

                time.sleep(
                    0.005
                )

                continue


            with self.lock:

                # Store only latest frame.
                self.frame = frame


    def latest(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()


    def stop(self):

        self.running = False


        try:

            self.thread.join(
                timeout=1.0
            )

        except Exception:
            pass


        try:

            self.cap.release()

        except Exception:
            pass


# =============================================================================
# ROI LOCATIONS
# =============================================================================

def get_roi_boxes(
    frame
):

    h, w = frame.shape[:2]


    # Move ROI slightly lower so timing/probability panels do not cover it.
    cx = w // 2
    cy = 305


    # -------------------------------------------------------------------------
    # Visible guide box
    # -------------------------------------------------------------------------

    gh = GUIDE_SIZE // 2


    gx0 = max(
        0,
        cx - gh
    )

    gy0 = max(
        0,
        cy - gh
    )

    gx1 = min(
        w,
        cx + gh
    )

    gy1 = min(
        h,
        cy + gh
    )


    # -------------------------------------------------------------------------
    # Larger hidden capture region
    # -------------------------------------------------------------------------

    ch = CAPTURE_SIZE // 2


    cx0 = max(
        0,
        cx - ch
    )

    cy0 = max(
        0,
        cy - ch
    )

    cx1 = min(
        w,
        cx + ch
    )

    cy1 = min(
        h,
        cy + ch
    )


    return (
        (gx0, gy0, gx1, gy1),
        (cx0, cy0, cx1, cy1)
    )


# =============================================================================
# FAST PREPROCESS
# =============================================================================

def preprocess_fast(
    frame
):
    """
    Fast MNIST-style preprocessing.

    The displayed square is 240x240.

    But we secretly capture 320x320.

    Therefore a stroke slightly outside/touching the visible square
    does not get chopped off.
    """

    guide_box, capture_box = (
        get_roi_boxes(frame)
    )


    x0, y0, x1, y1 = capture_box


    roi = frame[
        y0:y1,
        x0:x1
    ]


    # -------------------------------------------------------------------------
    # Grayscale
    # -------------------------------------------------------------------------

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    # -------------------------------------------------------------------------
    # Reduce before segmentation.
    #
    # 320x320 -> 112x112
    #
    # We do contour processing on only ~12k pixels.
    # -------------------------------------------------------------------------

    gray_small = cv2.resize(
        gray,
        (112, 112),
        interpolation=cv2.INTER_AREA
    )


    # Black pen / white paper -> MNIST polarity
    inv = cv2.bitwise_not(
        gray_small
    )


    # -------------------------------------------------------------------------
    # Otsu threshold
    # -------------------------------------------------------------------------

    _, mask = cv2.threshold(
        inv,
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU
    )


    # -------------------------------------------------------------------------
    # Find dominant digit stroke
    # -------------------------------------------------------------------------

    contour_result = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # Works on OpenCV 3 and OpenCV 4
    contours = contour_result[-2]


    digit28 = np.zeros(
        (
            28,
            28
        ),
        dtype=np.uint8
    )


    if len(contours) == 0:

        return (
            digit28,
            False,
            guide_box
        )


    cnt = max(
        contours,
        key=cv2.contourArea
    )


    area = cv2.contourArea(
        cnt
    )


    if area < 20:

        return (
            digit28,
            False,
            guide_box
        )


    x, y, w, h = cv2.boundingRect(
        cnt
    )


    # -------------------------------------------------------------------------
    # Padding around digit.
    # This further prevents clipping.
    # -------------------------------------------------------------------------

    pad = max(
        3,
        int(
            0.15
            *
            max(w, h)
        )
    )


    bx0 = max(
        0,
        x - pad
    )

    by0 = max(
        0,
        y - pad
    )

    bx1 = min(
        112,
        x + w + pad
    )

    by1 = min(
        112,
        y + h + pad
    )


    crop_gray = inv[
        by0:by1,
        bx0:bx1
    ].copy()


    crop_mask = mask[
        by0:by1,
        bx0:bx1
    ]


    # Remove background while keeping grayscale/antialiasing.
    crop_gray[
        crop_mask == 0
    ] = 0


    chh, cww = crop_gray.shape


    if chh == 0 or cww == 0:

        return (
            digit28,
            False,
            guide_box
        )


    # -------------------------------------------------------------------------
    # Fit digit inside 20x20 like MNIST
    # -------------------------------------------------------------------------

    scale = min(
        20.0 / float(chh),
        20.0 / float(cww)
    )


    new_h = max(
        1,
        int(
            round(
                chh * scale
            )
        )
    )


    new_w = max(
        1,
        int(
            round(
                cww * scale
            )
        )
    )


    resized = cv2.resize(
        crop_gray,
        (
            new_w,
            new_h
        ),
        interpolation=cv2.INTER_AREA
    )


    top = (
        28 - new_h
    ) // 2


    left = (
        28 - new_w
    ) // 2


    digit28[
        top:top + new_h,
        left:left + new_w
    ] = resized


    valid = (
        np.count_nonzero(
            digit28
        )
        > 15
    )


    return (
        digit28,
        valid,
        guide_box
    )


# =============================================================================
# SHARED DISPLAY STATE
# =============================================================================

state_lock = threading.Lock()


state = {
    "pred": 0,

    "confidence": 0.0,

    "probs": np.zeros(
        10,
        dtype=np.float64
    ),

    "digit28": np.zeros(
        (28, 28),
        dtype=np.uint8
    ),

    "inference_ms": 0.0,

    "conv1_ms": 0.0,

    "conv2_ms": 0.0,

    "preprocess_ms": 0.0,

    "fps": 0.0,

    "guide": None,

    "valid": False
}


# =============================================================================
# DRAW PROBABILITY PANEL
# =============================================================================

def draw_probability_panel(
    frame,
    probs
):

    h, w = frame.shape[:2]


    panel_x = w - 225
    panel_y = 5

    panel_w = 220
    panel_h = 170


    cv2.rectangle(
        frame,
        (
            panel_x,
            panel_y
        ),
        (
            panel_x + panel_w,
            panel_y + panel_h
        ),
        (0, 0, 0),
        -1
    )


    bar_x = panel_x + 25
    bar_width = 130


    for d in range(10):

        yy = (
            panel_y
            +
            14
            +
            d * 16
        )


        cv2.putText(
            frame,
            str(d),
            (
                panel_x + 6,
                yy + 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.rectangle(
            frame,
            (
                bar_x,
                yy - 5
            ),
            (
                bar_x + bar_width,
                yy + 5
            ),
            (130, 130, 130),
            1
        )


        filled = int(
            bar_width
            *
            float(probs[d])
        )


        if filled > 0:

            cv2.rectangle(
                frame,
                (
                    bar_x,
                    yy - 5
                ),
                (
                    bar_x + filled,
                    yy + 5
                ),
                (255, 255, 255),
                -1
            )


        cv2.putText(
            frame,
            "%.1f%%"
            % (
                float(probs[d])
                *
                100.0
            ),
            (
                bar_x
                +
                bar_width
                +
                5,
                yy + 4
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


# =============================================================================
# DRAW FRAME
# =============================================================================

def make_display_frame(
    frame,
    local_state
):

    out = frame.copy()


    # -------------------------------------------------------------------------
    # Timing panel
    # -------------------------------------------------------------------------

    cv2.rectangle(
        out,
        (5, 5),
        (400, 145),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        out,
        "Inference: %.1f ms"
        % local_state["inference_ms"],
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        out,
        "Live: %.1f FPS"
        % local_state["fps"],
        (215, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        out,
        "Conv1: %.3f ms"
        % local_state["conv1_ms"],
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        out,
        "Conv2 DMA sum: %.3f ms"
        % local_state["conv2_ms"],
        (175, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        out,
        "Preprocess: %.2f ms"
        % local_state["preprocess_ms"],
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # -------------------------------------------------------------------------
    # Probability panel
    # -------------------------------------------------------------------------

    draw_probability_panel(
        out,
        local_state["probs"]
    )


    # -------------------------------------------------------------------------
    # Guide box
    # -------------------------------------------------------------------------

    guide = local_state["guide"]


    if guide is not None:

        x0, y0, x1, y1 = guide


        cv2.rectangle(
            out,
            (
                x0,
                y0
            ),
            (
                x1,
                y1
            ),
            (255, 255, 255),
            3
        )


        # ---------------------------------------------------------------------
        # Prediction DIRECTLY ABOVE square
        # ---------------------------------------------------------------------

        if local_state["valid"]:

            text = (
                "Digit %d   %.1f%%"
                % (
                    local_state["pred"],
                    local_state["confidence"]
                )
            )

        else:

            text = "Place digit in box"


        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            2
        )[0]


        text_x = (
            x0
            +
            (
                (x1 - x0)
                -
                text_size[0]
            )
            // 2
        )


        text_y = max(
            170,
            y0 - 10
        )


        # Small black backing only behind text.
        cv2.rectangle(
            out,
            (
                text_x - 5,
                text_y
                -
                text_size[1]
                -
                6
            ),
            (
                text_x
                +
                text_size[0]
                +
                5,
                text_y + 5
            ),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            out,
            text,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    # -------------------------------------------------------------------------
    # 28x28 preview - bottom-left
    # -------------------------------------------------------------------------

    preview = cv2.resize(
        local_state["digit28"],
        (112, 112),
        interpolation=cv2.INTER_NEAREST
    )


    preview = cv2.cvtColor(
        preview,
        cv2.COLOR_GRAY2BGR
    )


    ph, pw = preview.shape[:2]


    px = 12
    py = out.shape[0] - ph - 12


    cv2.rectangle(
        out,
        (
            px - 3,
            py - 25
        ),
        (
            px + pw + 3,
            py + ph + 3
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        out,
        "28x28 input",
        (
            px,
            py - 7
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    out[
        py:py + ph,
        px:px + pw
    ] = preview


    cv2.rectangle(
        out,
        (
            px,
            py
        ),
        (
            px + pw,
            py + ph
        ),
        (255, 255, 255),
        1
    )


    return out


# =============================================================================
# START CAMERA
# =============================================================================

print("")
print("Starting Logitech C270...")


camera = CameraThread(
    CAMERA_INDEX
)


time.sleep(
    1.0
)


print("Camera started.")


# =============================================================================
# JUPYTER DISPLAY WIDGET
# =============================================================================

camera_widget = widgets.Image(
    format="jpeg",
    width=JPEG_WIDTH,
    height=JPEG_HEIGHT
)


display(
    camera_widget
)


# =============================================================================
# DISPLAY THREAD
# =============================================================================

display_running = True


def display_loop():

    global display_running


    while display_running:

        frame = camera.latest()


        if frame is None:

            time.sleep(
                0.01
            )

            continue


        with state_lock:

            local_state = {
                "pred":
                    state["pred"],

                "confidence":
                    state["confidence"],

                "probs":
                    state["probs"].copy(),

                "digit28":
                    state["digit28"].copy(),

                "inference_ms":
                    state["inference_ms"],

                "conv1_ms":
                    state["conv1_ms"],

                "conv2_ms":
                    state["conv2_ms"],

                "preprocess_ms":
                    state["preprocess_ms"],

                "fps":
                    state["fps"],

                "guide":
                    state["guide"],

                "valid":
                    state["valid"]
            }


        rendered = make_display_frame(
            frame,
            local_state
        )


        # Browser doesn't need full 640x480.
        rendered = cv2.resize(
            rendered,
            (
                JPEG_WIDTH,
                JPEG_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )


        ok, jpeg = cv2.imencode(
            ".jpg",
            rendered,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                JPEG_QUALITY
            ]
        )


        if ok:

            camera_widget.value = (
                jpeg.tobytes()
            )


        time.sleep(
            DISPLAY_INTERVAL
        )


display_thread = threading.Thread(
    target=display_loop
)


display_thread.daemon = True

display_thread.start()


# =============================================================================
# MAIN INFERENCE LOOP
# =============================================================================

print("")
print("Live inference running.")
print("Place black handwritten digit inside white square.")
print("Interrupt the notebook cell to stop.")
print("")


fps_counter = 0

fps_start = time.perf_counter()

live_fps = 0.0


try:

    while True:

        # ---------------------------------------------------------------------
        # Get newest camera frame
        # ---------------------------------------------------------------------

        frame = camera.latest()


        if frame is None:

            time.sleep(
                0.001
            )

            continue


        # ---------------------------------------------------------------------
        # Preprocess
        # ---------------------------------------------------------------------

        pre_t0 = time.perf_counter()


        (
            digit28,
            valid,
            guide
        ) = preprocess_fast(
            frame
        )


        preprocess_ms = (
            time.perf_counter()
            -
            pre_t0
        ) * 1000.0


        # Update preview even if digit isn't valid.
        with state_lock:

            state["digit28"] = (
                digit28.copy()
            )

            state["preprocess_ms"] = (
                preprocess_ms
            )

            state["guide"] = guide

            state["valid"] = valid


        if not valid:

            # Do NOT waste FPGA inference on blank frame.
            continue


        # ---------------------------------------------------------------------
        # FPGA inference
        # ---------------------------------------------------------------------

        (
            pred,
            confidence,
            probs,
            inference_ms,
            conv1_dma_ms,
            conv2_dma_sum_ms
        ) = infer_digit(
            digit28
        )


        # ---------------------------------------------------------------------
        # Classification FPS
        # ---------------------------------------------------------------------

        fps_counter += 1


        now = time.perf_counter()


        elapsed = (
            now
            -
            fps_start
        )


        if elapsed >= 1.0:

            live_fps = (
                fps_counter
                /
                elapsed
            )


            fps_counter = 0

            fps_start = now


        # ---------------------------------------------------------------------
        # Update state for display thread
        # ---------------------------------------------------------------------

        with state_lock:

            state["pred"] = pred

            state["confidence"] = (
                confidence
            )

            state["probs"] = (
                probs.copy()
            )

            state["inference_ms"] = (
                inference_ms
            )

            state["conv1_ms"] = (
                conv1_dma_ms
            )

            state["conv2_ms"] = (
                conv2_dma_sum_ms
            )

            state["fps"] = (
                live_fps
            )

            state["valid"] = True


except KeyboardInterrupt:

    print("")
    print("Stopping live classifier...")


finally:

    # -------------------------------------------------------------------------
    # Stop display thread
    # -------------------------------------------------------------------------

    display_running = False


    try:

        display_thread.join(
            timeout=1.0
        )

    except Exception:
        pass


    # -------------------------------------------------------------------------
    # Stop camera
    # -------------------------------------------------------------------------

    try:

        camera.stop()

    except Exception:
        pass


    # -------------------------------------------------------------------------
    # Ensure accelerator stopped
    # -------------------------------------------------------------------------

    try:

        engine.write(
            REG_CONTROL,
            0x00
        )

    except Exception:
        pass


    # -------------------------------------------------------------------------
    # Release DMA buffers
    # -------------------------------------------------------------------------

    try:
        conv1_in.freebuffer()
    except Exception:
        pass


    try:
        conv1_out_buf.freebuffer()
    except Exception:
        pass


    try:
        conv2_in.freebuffer()
    except Exception:
        pass


    try:
        conv2_out_buf.freebuffer()
    except Exception:
        pass


    print("Camera and DMA buffers released.")
