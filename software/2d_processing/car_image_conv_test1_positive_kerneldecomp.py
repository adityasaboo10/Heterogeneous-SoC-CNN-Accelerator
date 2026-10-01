import cv2
import numpy as np
import matplotlib.pyplot as plt
from pynq import Overlay, allocate
import time

print("Loading Overlay...")
overlay = Overlay("design_1.bit")
dma = overlay.axi_dma_0
engine = overlay.Vector_Engine_IP

# --- THE UNIVERSAL POSITIVE KERNEL PACKER ---
def load_positive_kernel(engine, kernel_matrix):
    """Flips the matrix and packs POSITIVE ONLY weights to AXI-Lite"""
    flipped = np.flipud(kernel_matrix).flatten()
    
    # Cast to standard integers (Hardware expects unsigned)
    k = [int(val) for val in flipped]
    
    # Pack 4 weights per 32-bit register (Little Endian)
    reg0 = (k[3] << 24) | (k[2] << 16) | (k[1] << 8) | k[0]
    reg1 = (k[7] << 24) | (k[6] << 16) | (k[5] << 8) | k[4]
    reg2 = k[8]
    
    engine.write(0x00, reg0)
    engine.write(0x04, reg1)
    engine.write(0x08, reg2)

# --- 1. DEFINE THE SIGNED KERNEL ---
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Mathematically split the kernel into two strictly positive matrices
w_pos = np.maximum(edge_kernel, 0)
w_neg = np.maximum(-edge_kernel, 0)

# --- 2. PREPARE THE IMAGE ---
image_path = 'car.jpg' # Ensure your photo is uploaded
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (256, 256))

# Allocate DMA buffers
input_buffer = allocate(shape=(65536,), dtype=np.uint8)
output_buffer = allocate(shape=(64516,), dtype=np.uint32) 

np.copyto(input_buffer, img_resized.flatten())
engine.write(0x3C, 64516) # Lock the tlast wrapper

print("\n--- STARTING HARDWARE ACCELERATION ---")
# Start the Wall-Clock Timer!
system_start_time = time.perf_counter()

# --- 3. PASS 1: THE POSITIVE WEIGHTS ---
load_positive_kernel(engine, w_pos)
engine.write(0x30, 0x00000001) # Start

dma.recvchannel.transfer(output_buffer) 
dma.sendchannel.transfer(input_buffer)  
dma.sendchannel.wait()
dma.recvchannel.wait()

# Read stopwatch before resetting
cycles_pass1 = engine.read(0x34)
engine.write(0x30, 0x00000000) # Reset engine

# Save the results of Pass 1 and convert to signed 32-bit integer for safe subtraction
result_pos = np.copy(output_buffer).astype(np.int32)

# --- 4. PASS 2: THE NEGATIVE WEIGHTS ---
load_positive_kernel(engine, w_neg)
engine.write(0x30, 0x00000001) # Start

dma.recvchannel.transfer(output_buffer) 
dma.sendchannel.transfer(input_buffer)  
dma.sendchannel.wait()
dma.recvchannel.wait()

# Read stopwatch before resetting
cycles_pass2 = engine.read(0x34)
engine.write(0x30, 0x00000000) # Reset engine

result_neg = np.copy(output_buffer).astype(np.int32)

# --- 5. RECONSTRUCT IN SOFTWARE ---
# The magic math: Subtract the negative pass from the positive pass!
final_flat_output = result_pos - result_neg

# Reshape to 2D
hardware_output = final_flat_output.reshape((254, 254))

# Clip the values to standard visual ranges (0 to 255) to remove pitch-black artifacts
hardware_output_clipped = np.clip(hardware_output, 0, 255).astype(np.uint8)

# Stop the Wall-Clock Timer!
system_end_time = time.perf_counter()

# --- CALCULATE METRICS ---
# Convert cycles to milliseconds (assuming 50MHz clock = 20ns period)
total_cycles = cycles_pass1 + cycles_pass2
silicon_time_ms = ((total_cycles * 20) / 1000) / 1000 

# System time includes Python overhead, DMA setup, and matrix reconstruction
system_time_ms = (system_end_time - system_start_time) * 1000
estimated_fps = 1000 / system_time_ms

print("\n--- BENCHMARK RESULTS ---")
print(f"Total Silicon Math Time (Both Passes): {silicon_time_ms:.2f} ms")
print(f"Full System Time (Passes + DMA + Python): {system_time_ms:.2f} ms")
print(f"Estimated Pipeline Speed: {estimated_fps:.1f} FPS")

# --- 6. DISPLAY ---
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_resized, cmap='gray')
ax[0].set_title('Original Input (256x256)')
ax[0].axis('off')

ax[1].imshow(hardware_output_clipped, cmap='gray')
ax[1].set_title('FPGA Edge Detection (Two-Pass)')
ax[1].axis('off')

plt.show()
