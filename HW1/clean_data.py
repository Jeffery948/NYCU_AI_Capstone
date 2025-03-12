import os
import random
from PIL import Image

# 原始資料夾
real_dir = "raw_data/real_animal_images"
ai_dir = "raw_data/ai_animal_images"

# 目標資料夾
train_real_dir = "clean_data/train/real"
train_ai_dir = "clean_data/train/ai"
test_real_dir = "clean_data/test/real"
test_ai_dir = "clean_data/test/ai"

# 確保目標資料夾存在
for folder in [train_real_dir, train_ai_dir, test_real_dir, test_ai_dir]:
    os.makedirs(folder, exist_ok=True)

# 取得所有圖片
real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
ai_images = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir)]

# 隨機打亂圖片
random.shuffle(real_images)
random.shuffle(ai_images)

# 分割training/testing set
train_real = real_images[:225]
test_real = real_images[225:300]
train_ai = ai_images[:225]
test_ai = ai_images[225:300]

def process_and_save(images, output_dir):
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img_name = os.path.basename(img_path)
        img.save(os.path.join(output_dir, img_name))

# 移動圖片到對應的資料夾
process_and_save(train_real, train_real_dir)
process_and_save(test_real, test_real_dir)
process_and_save(train_ai, train_ai_dir)
process_and_save(test_ai, test_ai_dir)

print(f"Training set: Real: {len(train_real)}, AI: {len(train_ai)}")
print(f"Testing set: Real: {len(test_real)}, AI: {len(test_ai)}")