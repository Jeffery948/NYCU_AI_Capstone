import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests

# 設定圖片存放資料夾
SAVE_DIR = "raw_data/real_animal_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 啟動 Chrome 瀏覽器
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # 不開啟瀏覽器視窗
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920x1080")
driver = webdriver.Chrome(service=service, options=options)

# 目標 URL
URL = "https://unsplash.com/s/photos/animal?license=free"
driver.get(URL)

# 模擬滾動，確保載入更多圖片
click = False
for _ in range(50):
    if not click:
        try:
            # 嘗試尋找並點擊 "Load More" 按鈕
            load_more_button = driver.find_element(By.XPATH, "//button[contains(text(),'Load more')]")
            load_more_button.click()  # 點擊 "Load More" 按鈕
            time.sleep(2)  # 等待圖片加載
            click = True
            print("Click success!")
        except Exception as e:
            print("No more 'Load More' button or failed to click:", e)
    driver.execute_script("window.scrollBy(0, 1000);")
    time.sleep(2)

# 抓取所有圖片 URL
images = driver.find_elements(By.TAG_NAME, "img")
image_urls = [img.get_attribute("src") for img in images if img.get_attribute("src")]

# 過濾掉 base64 和廣告圖片
filtered_urls = [url for url in image_urls if url and not url.startswith("data:image") and "https://images.unsplash.com/photo" in url]
print(len(filtered_urls))

# 關閉瀏覽器
driver.quit()

# 下載圖片
for i, img_url in enumerate(filtered_urls[:300]):  # 限制下載 300 張
    try:
        img_data = requests.get(img_url).content
        with open(os.path.join(SAVE_DIR, f"{i+1}.jpg"), "wb") as f:
            f.write(img_data)
        print(f"Downloaded: animal_{i+1}.jpg")
    except Exception as e:
        print(f"Failed to download image {i+1}: {e}")

print("所有圖片下載完成！")
