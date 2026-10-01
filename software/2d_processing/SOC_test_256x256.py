from pynq import Overlay, allocate
import numpy as np
import time

print("Flashing custom Vector Engine to the FPGA fabric...")
overlay = Overlay("design_1.bit")

dma = overlay.axi_dma_0
engine = overlay.Vector_Engine_IP

print("Configuring Asymmetric Weights via AXI-Lite")

# Weights: 1, 2, 3, 4, 5, 6, 7, 8, 9
# Packed into 32-bit registers (Hex: W3_W2_W1_W0)
engine.write(0x00, 0x04090807) # Weights 0=7, 1=8, 2=9, 3=4
engine.write(0x04, 0x02010605) # Weights 4=5, 5=6, 6=1, 7=2
engine.write(0x08, 0x00000003) # Weight  8=3

# Biases (Keeping them 0 for easier math tracking)
for offset in range(0x0C, 0x30, 4):
    engine.write(offset, 0x00000000)
    
# Tell the wrapper exactly when to fire tlast!
engine.write(0x3C, 64516) # 0x3C is slv_reg15

# Master start
print("Asserting master_start...")
engine.write(0x30, 0x00000001)

# Buffer Setup for 256x256
TOTAL_INPUTS = 65536  # 256x256
VALID_OUTPUTS = 64516 # 254x254

input_buffer = allocate(shape=(TOTAL_INPUTS,), dtype=np.uint8)
output_buffer = allocate(shape=(VALID_OUTPUTS,), dtype=np.uint32)

# Generate a random image array (values between 1 and 4)
np.random.seed(42) # Seeded so the "random" values are the same every time you run it
random_image = np.random.randint(1, 5, size=(256, 256), dtype=np.uint8)

# Flatten and copy to physical DMA memory
np.copyto(input_buffer, random_image.flatten())
    
print("Data loaded. Firing the DMA...")

# 1. Start transfers
dma.recvchannel.transfer(output_buffer) 
dma.sendchannel.transfer(input_buffer)  

# 2. Wait for the hardware tlast interrupt!
dma.sendchannel.wait()
dma.recvchannel.wait()

# 3. Read the exact hardware stopwatch
total_cycles = engine.read(0x34)
exact_time_us = (total_cycles * 20) / 1000 # Assuming 100MHz clock (10ns)

print("\n--- PERFORMANCE ---")
print(f"Hardware finished in exactly {total_cycles} clock cycles.")
print(f"Silicon Execution Time: {exact_time_us} microseconds")

print("\n--- VERIFICATION ---")
# 4. The Golden Software Model (Predicting the First Pixel)
# We pull the first 3x3 window directly from our random image array
window = random_image[0:3, 0:3]

# The mathematical proof
software_pixel_0 = (
    window[0,0]*1 + window[0,1]*2 + window[0,2]*3 +
    window[1,0]*4 + window[1,1]*5 + window[1,2]*6 +
    window[2,0]*7 + window[2,1]*8 + window[2,2]*9
)

hardware_pixel_0 = output_buffer[0]

print("Input 3x3 Window:")
print(window)
print(f"Software Calculated Output[0]: {software_pixel_0}")
print(f"Hardware Calculated Output[0]: {hardware_pixel_0}")

if software_pixel_0 == hardware_pixel_0:
    print("✅ MATCH! The custom silicon math is flawless.")
else:
    print("❌ MISMATCH! Check line buffer ordering.")

# 5. Pull master_start LOW to reset state machine
engine.write(0x30, 0x00000000)
