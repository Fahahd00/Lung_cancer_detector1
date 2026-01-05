import os
import numpy as np
from PIL import Image, ImageFilter
import shutil

IMG_DIR = "img"

# Remove old images
if os.path.exists(IMG_DIR):
    shutil.rmtree(IMG_DIR)
os.makedirs(IMG_DIR, exist_ok=True)

print("Creating lung image database...")

np.random.seed(42)

# ========== HEALTHY IMAGES ==========
print("\nGenerating HEALTHY images (no spots)...")

for i in range(15):
    # Create base image
    arr = np.ones((224, 224), dtype=np.uint8) * 150
    
    # Add gradient from center
    for x in range(224):
        for y in range(224):
            dist = np.sqrt((x-112)**2 + (y-112)**2)
            intensity = int(150 - (dist / 160) * 50)
            intensity = np.clip(intensity, 80, 200)
            
            # Add slight noise
            noise = np.random.randint(-5, 5)
            arr[x, y] = intensity + noise
    
    # Smooth the image
    img = Image.fromarray(arr, mode='L')
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img.save(f'{IMG_DIR}/normal_{i+1}.jpg')
    print(f"  Created normal_{i+1}.jpg")

# ========== CANCER IMAGES ==========
print("\nGenerating CANCER images (with dark spots)...")

for i in range(15):
    # Create base image
    arr = np.ones((224, 224), dtype=np.uint8) * 150
    
    # Add gradient from center
    for x in range(224):
        for y in range(224):
            dist = np.sqrt((x-112)**2 + (y-112)**2)
            intensity = int(150 - (dist / 160) * 50)
            intensity = np.clip(intensity, 80, 200)
            
            # Add slight noise
            noise = np.random.randint(-5, 5)
            arr[x, y] = intensity + noise
    
    # Add dark spots (tumors)
    num_tumors = np.random.randint(2, 5)
    for _ in range(num_tumors):
        center_x = np.random.randint(50, 174)
        center_y = np.random.randint(50, 174)
        radius = np.random.randint(15, 35)
        
        # Draw dark spot
        for x in range(max(0, center_x-radius), min(224, center_x+radius)):
            for y in range(max(0, center_y-radius), min(224, center_y+radius)):
                if (x-center_x)**2 + (y-center_y)**2 <= radius**2:
                    arr[x, y] = int(arr[x, y] * 0.4)  # Darken by 60%
    
    # Smooth the image
    img = Image.fromarray(arr, mode='L')
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img.save(f'{IMG_DIR}/cancer_{i+1}.jpg')
    print(f"  Created cancer_{i+1}.jpg")

print("\n✓ Database created successfully!")
print("  - 15 HEALTHY images (uniform, no dark spots)")
print("  - 15 CANCER images (with dark spots = tumors)")
