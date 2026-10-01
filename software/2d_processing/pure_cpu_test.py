import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Load and Resize Image (Identical to the hardware setup)
image_path = 'car.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (256, 256))

# 2. Define the exact same Edge Detection Kernel
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

print("--- STARTING PURE CPU (PROCESSING SYSTEM) EXECUTION ---")
# Track time in nanoseconds for exact precision match to hardware
cpu_start = time.perf_counter_ns()

# Run the 2D convolution directly on the ARM processor
cpu_output = cv2.filter2D(img_resized, ddepth=cv2.CV_32F, kernel=edge_kernel)

cpu_end = time.perf_counter_ns()

# --- METRICS CALCULATIONS ---
cpu_latency_ms = (cpu_end - cpu_start) / 1_000_000
estimated_fps = 1000 / cpu_latency_ms

print("\n--- BENCHMARK RESULTS ---")
print(f"Pure CPU Latency: {cpu_latency_ms:.4f} ms")
print(f"Estimated CPU Speed: {estimated_fps:.1f} FPS")

# --- VISUAL RECONSTRUCTION ---
# Clip values to standard visual range (0 to 255) for rendering
cpu_output_clipped = np.clip(cpu_output, 0, 255).astype(np.uint8)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_resized, cmap='gray')
ax[0].set_title('Original Input (256x256)')
ax[0].axis('off')

ax[1].imshow(cpu_output_clipped, cmap='gray')
ax[1].set_title(f'CPU Output ({cpu_latency_ms:.2f}ms)')
ax[1].axis('off')

plt.show()
