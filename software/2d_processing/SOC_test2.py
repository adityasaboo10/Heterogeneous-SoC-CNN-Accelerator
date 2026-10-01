from pynq import Overlay, allocate
import numpy as np
import time

print("Flashing custom Vector Engine to the FPGA fabric...")
overlay = Overlay("conv_engine.bit")

dma = overlay.axi_dma_0
engine = overlay.Vector_Engine_IP

print("Config Weights and Biases via AXI-Lite")

# Weights
engine.write(0x00, 0x01010101) # Weights 0, 1, 2, 3
engine.write(0x04, 0x01010101) # Weights 4, 5, 6, 7
engine.write(0x08, 0x00000001) # Weight 8

# Biases
for offset in range(0x0C, 0x30, 4):
    engine.write(offset, 0x00000000)
   
# Master start
print("Asserting master_start...")
engine.write(0x30, 0x00000001)

# defining PIXELS - THESE WERE THE MISSING LINES!
TOTAL_INPUTS = 400 # 20x20
VALID_OUTPUTS = 324 # 18x18

# Allocate different sizes!
input_buffer = allocate(shape=(TOTAL_INPUTS,), dtype=np.uint8)
output_buffer = allocate(shape=(VALID_OUTPUTS,), dtype=np.uint32)

# Fill the input buffer with test data (e.g., all 2s)
for i in range(TOTAL_INPUTS):
    input_buffer[i] = 2

# CRITICAL: Pre-fill the output buffer with 0s so we know it's empty
for i in range(VALID_OUTPUTS):
    output_buffer[i] = 0
    
print("Data loaded. Firing the DMA...")

# Start the clock exactly as the firehose opens!
start_time = time.time()

# 1. Start transfers
dma.recvchannel.transfer(output_buffer) 
dma.sendchannel.transfer(input_buffer)  

# 2. THE RAM POLLING HACK
# Force Python into a tight loop checking the very last pixel in the array.
# VALID_OUTPUTS is 324, so the last index is 323.
while output_buffer[323] == 0:
    pass 

# Stop the clock the exact moment the loop breaks!
end_time = time.time()

print(f"Hardware computed 324 pixels in: {end_time - start_time:.6f} seconds!")

# 3. Print the RAM directly
print("First 20 MAC Results:")
print(output_buffer[:20])

# 4. Pull master_start LOW to reset state machine for next run
engine.write(0x30, 0x00000000)
