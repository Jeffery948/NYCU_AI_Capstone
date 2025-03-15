import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests

# Set the folder for saving images
SAVE_DIR = "raw_data/real_animal_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Start Chrome browser
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run without opening a browser window
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920x1080")
driver = webdriver.Chrome(service=service, options=options)

# Target URL
URL = "https://unsplash.com/s/photos/animal?license=free"
driver.get(URL)

# Simulate scrolling to ensure more images are loaded
click = False
for _ in range(50):
    if not click:
        try:
            # Try to find and click the "Load More" button
            load_more_button = driver.find_element(By.XPATH, "//button[contains(text(),'Load more')]")
            load_more_button.click()  # Click the "Load More" button
            time.sleep(2)  # Wait for images to load
            click = True
            print("Click success!")
        except Exception as e:
            print("No more 'Load More' button or failed to click:", e)
    driver.execute_script("window.scrollBy(0, 1000);")
    time.sleep(2)

# Grab all image URLs
images = driver.find_elements(By.TAG_NAME, "img")
image_urls = [img.get_attribute("src") for img in images if img.get_attribute("src")]

# Filter out base64 and advertisement images
filtered_urls = [url for url in image_urls if url and not url.startswith("data:image") and "https://images.unsplash.com/photo" in url]
print(len(filtered_urls))

# Close the browser
driver.quit()

# Download images
for i, img_url in enumerate(filtered_urls[:300]):  # Limit to downloading 300 images
    try:
        img_data = requests.get(img_url).content
        with open(os.path.join(SAVE_DIR, f"{i+1}.jpg"), "wb") as f:
            f.write(img_data)
        print(f"Downloaded: animal_{i+1}.jpg")
    except Exception as e:
        print(f"Failed to download image {i+1}: {e}")

print("All images have been downloaded!")