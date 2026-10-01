import cv2
import numpy as np
from pynq import Overlay, allocate
from IPython.display import display, Image, clear_output
import time

print("Loading Hardware Overlay...")
overlay = Overlay("design_1.bit")

dmas = [overlay.axi_dma_0, overlay.axi_dma_1, overlay.axi_dma_2]
engines = [overlay.Vector_Engine_IP_0, overlay.Vector_Engine_IP_1, overlay.Vector_Engine_IP_2]

edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

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

# --- Write weights ONCE (kernel doesn't change frame to frame) ---
print("Writing weights to all 3 engines...")
for engine in engines:
    write_weights_and_biases(engine, edge_kernel)

# --- Allocate DMA buffers ONCE ---
input_buffers = [allocate(shape=(65536,), dtype=np.uint8) for _ in range(3)]
output_buffers = [allocate(shape=(64516,), dtype=np.int32) for _ in range(3)]

def run_rgb_convolution(bgr_frame):
    resized = cv2.resize(bgr_frame, (256, 256))
    b, g, r = cv2.split(resized)
    channels = [r, g, b]  # R->engine0, G->engine1, B->engine2

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

    outputs = [np.clip(np.copy(output_buffers[i]).reshape((254, 254)), 0, 255).astype(np.uint8)
               for i in range(3)]
    r_out, g_out, b_out = outputs
    merged_bgr = cv2.merge([b_out, g_out, r_out])
    return merged_bgr, hw_time_ms

# --- Webcam setup ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
else:
    print("Running live RGB FPGA edge detection... Interrupt the kernel (Stop button) to end.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed")
                break

            result, hw_time_ms = run_rgb_convolution(frame)

            # --- Build side-by-side display ---
            original_resized = cv2.resize(frame, (256, 256))
            output_resized = cv2.resize(result, (256, 256))  # match dims for side-by-side
            combined = np.hstack((original_resized, output_resized))

            cv2.putText(combined, f"HW: {hw_time_ms:.2f} ms", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', combined)
            clear_output(wait=True)
            display(Image(data=jpeg.tobytes()))

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        cap.release()
        for buf in input_buffers + output_buffers:
            buf.freebuffer()
        print("Buffers freed, camera released.")
