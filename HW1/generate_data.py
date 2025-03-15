import requests
import os
import base64

# Stable Diffusion API endpoint
url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

SAVE_DIR = "raw_data/ai_animal_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Read prompts
with open('prompts.txt', 'r') as f:
    prompts = f.readlines()

with open('negative prompt.txt', 'r') as ff:
    negative_prompts = ff.readline()

# Generate images
for i, prompt in enumerate(prompts):
    payload = {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompts.strip(),
        "steps": 50,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "sampler_index": "DPM++ 2M SDE",
        "scheduler": "Align Your Steps",
        "seed": -1
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        img_data = response.json().get("images")[0]  # The returned image is base64 encoded
        # Decode base64 and save as a file
        img_bytes = base64.b64decode(img_data)
        img_path = os.path.join(SAVE_DIR, f"{i+1}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        print(f"Generated: {img_path}")
    else:
        print(f"Failed to generate image {i+1}")
        print(response.status_code)

print("AI-generated animal images are complete!")