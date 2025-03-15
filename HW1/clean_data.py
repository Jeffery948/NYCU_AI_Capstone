import os
import random
from PIL import Image

# Original data folder
real_dir = "raw_data/real_animal_images"
ai_dir = "raw_data/ai_animal_images"

# Target folders
train_real_dir = "clean_data/train/real"
train_ai_dir = "clean_data/train/ai"
test_real_dir = "clean_data/test/real"
test_ai_dir = "clean_data/test/ai"

# Ensure target folders exist
for folder in [train_real_dir, train_ai_dir, test_real_dir, test_ai_dir]:
    os.makedirs(folder, exist_ok=True)

# Get all image paths
real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
ai_images = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir)]

# Shuffle the images randomly
random.shuffle(real_images)
random.shuffle(ai_images)

# Split into training/testing sets
train_real = real_images[:225]
test_real = real_images[225:300]
train_ai = ai_images[:225]
test_ai = ai_images[225:300]

def process_and_save(images, output_dir):
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img_name = os.path.basename(img_path)
        img.save(os.path.join(output_dir, img_name))

# Move images to corresponding folders
process_and_save(train_real, train_real_dir)
process_and_save(test_real, test_real_dir)
process_and_save(train_ai, train_ai_dir)
process_and_save(test_ai, test_ai_dir)

print(f"Training set: Real: {len(train_real)}, AI: {len(train_ai)}")
print(f"Testing set: Real: {len(test_real)}, AI: {len(test_ai)}")