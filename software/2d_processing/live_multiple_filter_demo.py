import cv2
import numpy as np
from pynq import Overlay, allocate
import ipywidgets as widgets
from IPython.display import display, Image, clear_output
import asyncio
import time

print("Loading Hardware Overlay...")
overlay = Overlay("design_1.bit")

dmas = [overlay.axi_dma_0, overlay.axi_dma_1, overlay.axi_dma_2]
engines = [overlay.Vector_Engine_IP_0, overlay.Vector_Engine_IP_1, overlay.Vector_Engine_IP_2]

# --- Filter presets: (kernel, software scale, software offset) ---
FILTERS = {
    "Identity":  (np.array([[0,0,0],[0,1,0],[0,0,0]]), 1.0, 0),
    "Edge Detect": (np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]), 1.0, 0),
    "Sharpen":   (np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]), 1.0, 0),
    "Emboss":    (np.array([[-2,-1,0],[-1,1,1],[0,1,2]]), 1.0, 128),
    "Blur":      (np.array([[1,1,1],[1,1,1],[1,1,1]]), 1.0/9.0, 0),
}

current_filter_name = "Edge Detect"

def write_weights_and_biases(engine, weights, biases=None):
    flat_weights = weights.flatten()
    k = [int(val) & 0xFF for val in flat_weights]
    reg0 = (k[3] << 24) | (k[2] << 16) | (k[1] << 8) | k[0]
    reg1 = (k[7] << 24) | (k[6] << 16) | (k[5] << 8) | k[4]
    reg2 = k[8]
    engine.write(0x00, reg0)
    engine.write(0x04, reg1)
    engine.write(0x08, reg2)
    if biases is None:
        biases = [0] * 9
    else:
        biases = biases.flatten()
    for i in range(9):
        engine.write(0x0C + (i * 4), int(biases[i]) & 0xFFFFFFFF)

def apply_filter(name):
    """Push a new kernel to all 3 engines."""
    global current_filter_name
    kernel, scale, offset = FILTERS[name]
    for engine in engines:
        write_weights_and_biases(engine, kernel)
    current_filter_name = name
    print(f"Switched to: {name}")

# --- Write initial filter ---
apply_filter(current_filter_name)

# --- DMA buffers, allocated once ---
input_buffers = [allocate(shape=(65536,), dtype=np.uint8) for _ in range(3)]
output_buffers = [allocate(shape=(64516,), dtype=np.int32) for _ in range(3)]

def run_rgb_convolution(bgr_frame):
    resized = cv2.resize(bgr_frame, (256, 256))
    b, g, r = cv2.split(resized)
    channels = [r, g, b]

    hw_start = time.perf_counter_ns()
    for i, ch in enumerate(channels):
        np.copyto(input_buffers[i], ch.flatten())
        engines[i].write(0x3C, 64516)
        engines[i].write(0x30, 0x00000001)
        dmas[i].recvchannel.transfer(output_buffers[i])
        dmas[i].sendchannel.transfer(input_buffers[i])

    for i in range(3):
        dmas[i].sendchannel.wait()
        dmas[i].recvchannel.wait()
        engines[i].write(0x30, 0x00000000)
    hw_end = time.perf_counter_ns()
    hw_time_ms = (hw_end - hw_start) / 1_000_000

    # --- Software scale/offset (handles Blur/Emboss normalization) ---
    _, scale, offset = FILTERS[current_filter_name]
    outputs = []
    for i in range(3):
        raw = np.copy(output_buffers[i]).reshape((254, 254)).astype(np.float32)
        adjusted = raw * scale + offset
        outputs.append(np.clip(adjusted, 0, 255).astype(np.uint8))

    r_out, g_out, b_out = outputs
    merged_bgr = cv2.merge([b_out, g_out, r_out])
    return merged_bgr, hw_time_ms

# ==========================================
# --- UI: buttons + video output area ---
# ==========================================
video_out = widgets.Output()
buttons = []

def make_on_click(name):
    def handler(b):
        apply_filter(name)
    return handler

for name in FILTERS:
    btn = widgets.Button(description=name)
    btn.on_click(make_on_click(name))
    buttons.append(btn)

stop_button = widgets.Button(description="Stop", button_style="danger")
button_row = widgets.HBox(buttons + [stop_button])

display(button_row)
display(video_out)

running = True
def stop_handler(b):
    global running
    running = False
stop_button.on_click(stop_handler)

# --- Async live loop so button clicks get processed ---
async def live_loop():
    global running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with video_out:
            print("ERROR: Could not open webcam.")
        return

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            result, hw_time_ms = run_rgb_convolution(frame)

            original_resized = cv2.resize(frame, (256, 256))
            output_resized = cv2.resize(result, (256, 256))
            combined = np.hstack((original_resized, output_resized))

            cv2.putText(combined, f"{current_filter_name} | HW: {hw_time_ms:.2f} ms",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', combined)
            with video_out:
                clear_output(wait=True)
                display(Image(data=jpeg.tobytes()))

            await asyncio.sleep(0.001)  # yields control so button clicks register

    finally:
        cap.release()
        for buf in input_buffers + output_buffers:
            buf.freebuffer()
        with video_out:
            print("Stopped. Camera released, buffers freed.")

asyncio.create_task(live_loop())
