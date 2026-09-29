import cv2
import numpy as np
import matplotlib.pyplot as plt
from pynq import Overlay, allocate
import time

print("Loading Hardware Overlay...")
# Make sure your newly synthesized bitstream is uploaded!
overlay = Overlay("design_1.bit")
dma = overlay.axi_dma_0
engine = overlay.Vector_Engine_IP

def run_hardware_convolution(engine, dma, image, weights, biases=None):
    """
    Executes a 3x3 convolution on the custom FPGA Vector Engine.
    """
    # 1. --- FORMAT AND PACK WEIGHTS ---
    flat_weights = weights.flatten()
    
    # int(val) remove the int8,fp32 from numpy, & with 0xFF keeps 8 bits
    k = [int(val) & 0xFF for val in flat_weights]
    
    # Pack four 8-bit weights into each 32-bit register (Little Endian)
    reg0 = (k[3] << 24) | (k[2] << 16) | (k[1] << 8) | k[0]
    reg1 = (k[7] << 24) | (k[6] << 16) | (k[5] << 8) | k[4]
    reg2 = k[8]
    
    print("Writing Weights")
    engine.write(0x00, reg0)
    engine.write(0x04, reg1)
    engine.write(0x08, reg2)
    print("written")
    
    # 2. --- FORMAT AND PACK BIASES ---
    if biases is None:
        biases = [0] * 9 # Default to 0 if no biases are provided
    else:
        biases = biases.flatten()
        
    for i in range(9):
        # Biases are 32-bit, starting at memory offset 0x0C
        b_val = int(biases[i]) & 0xFFFFFFFF
        engine.write(0x0C + (i * 4), b_val)
        
    # 3. --- PREPARE IMAGE & ALLOCATE DMA MEMORY ---
    img_resized = cv2.resize(image, (256, 256))
    
    input_buffer = allocate(shape=(65536,), dtype=np.uint8)
    output_buffer = allocate(shape=(64516,), dtype=np.int32)
    
    np.copyto(input_buffer, img_resized.flatten())
    
    # 4. --- FIRE THE HARDWARE ---
    engine.write(0x3C, 64516)      # Lock TLAST for 254x254 outpu
    engine.write(0x30, 0x00000001) # Assert master_start(32 bits, send high signal)
    #-------------------------
    hw_start = time.perf_counter_ns()
    
    dma.recvchannel.transfer(output_buffer) 
    dma.sendchannel.transfer(input_buffer)  
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    
    # --- STOP PURE HARDWARE TIMER ---
    hw_end = time.perf_counter_ns()
    
    engine.write(0x30, 0x00000000) # De-assert master_start
    hardware_cycles = engine.read(0x34)
    # Copy the hardware output to a standard numpy array
    
    hardware_output = np.copy(output_buffer).reshape((254, 254))
    
    # Free the physical memory to prevent SoC crashes
    input_buffer.freebuffer()
    output_buffer.freebuffer()
    
    exact_hw_time_ms = (hw_end - hw_start) / 1_000_000
    
    return hardware_output, exact_hw_time_ms

# --- MAIN EXECUTION & BENCHMARKING ---
# ==========================================

edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

image_path = 'car.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

print("\n--- STARTING SINGLE-PASS ACCELERATION ---")
system_start_time = time.perf_counter()

result, silicon_time_ms = run_hardware_convolution(engine, dma, img, edge_kernel)
system_end_time = time.perf_counter()

# --- METRICS CALCULATIONS ---
system_time_ms = (system_end_time - system_start_time) * 1000
estimated_fps = 1000 / system_time_ms

print("\n--- BENCHMARK RESULTS ---")
print(f"Hardware Latency (DMA + Silicon): {silicon_time_ms:.2f} ms")
print(f"Full System Time (OpenCV + HW + Python): {system_time_ms:.2f} ms")
print(f"Estimated Pipeline Speed: {estimated_fps:.1f} FPS")

#---------Visual Construction--------
# Hardware mathematically allows massive/negative numbers. Clip them for your monitor.
hardware_output_clipped = np.clip(result, 0, 255).astype(np.uint8)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.resize(img, (256, 256)), cmap='gray')
ax[0].set_title('Original Input (256x256)')
ax[0].axis('off')

ax[1].imshow(hardware_output_clipped, cmap='gray')
ax[1].set_title(f'FPGA Edge Detection ({silicon_time_ms:.2f}ms)')
ax[1].axis('off')

plt.show()

