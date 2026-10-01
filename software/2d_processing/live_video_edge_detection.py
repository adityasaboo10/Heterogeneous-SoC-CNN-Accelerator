import cv2
import numpy as np
from pynq import Overlay, allocate
from IPython.display import display, Image, clear_output
import time

print("Loading Hardware Overlay...")
overlay = Overlay("design_1.bit")
dma = overlay.axi_dma_0
engine = overlay.Vector_Engine_IP

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
        b_val = int(biases[i]) & 0xFFFFFFFF
        engine.write(0x0C + (i * 4), b_val)

# --- Write weights ONCE (kernel doesn't change frame to frame) ---
write_weights_and_biases(engine, edge_kernel)

# --- Allocate DMA buffers ONCE (V2.2 zero-overhead optimization) ---
input_buffer = allocate(shape=(65536,), dtype=np.uint8)
output_buffer = allocate(shape=(64516,), dtype=np.int32)

def run_hardware_convolution_live(gray_frame):
    img_resized = cv2.resize(gray_frame, (256, 256))
    np.copyto(input_buffer, img_resized.flatten())

    engine.write(0x3C, 64516)
    engine.write(0x30, 0x00000001)

    hw_start = time.perf_counter_ns()
    dma.recvchannel.transfer(output_buffer)
    dma.sendchannel.transfer(input_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    hw_end = time.perf_counter_ns()

    engine.write(0x30, 0x00000000)

    hardware_output = np.copy(output_buffer).reshape((254, 254))
    exact_hw_time_ms = (hw_end - hw_start) / 1_000_000
    return hardware_output, exact_hw_time_ms

# --- Webcam setup ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
else:
    print("Webcam opened. Running live FPGA edge detection... Interrupt the kernel to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Run through FPGA ---
            result, hw_time_ms = run_hardware_convolution_live(gray)
            output_clipped = np.clip(result, 0, 255).astype(np.uint8)

            # --- Build side-by-side display ---
            original_resized = cv2.resize(gray, (256, 256))
            output_resized = cv2.resize(output_clipped, (256, 256))
            combined = np.hstack((original_resized, output_resized))

            # Overlay HW latency text
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            cv2.putText(combined_bgr, f"HW: {hw_time_ms:.2f} ms", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            _, jpeg = cv2.imencode('.jpg', combined_bgr)
            clear_output(wait=True)
            display(Image(data=jpeg.tobytes()))

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        cap.release()
        input_buffer.freebuffer()
        output_buffer.freebuffer()
        print("Buffers freed, camera released.")
