import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# 設定 HOG 參數
hog_params = dict(
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    orientations=9,
    block_norm='L2-Hys'
)

def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 讀取灰階
    image = cv2.resize(image, (256, 256))  # 調整大小
    features = hog(image, **hog_params)
    return features

# 讀取資料
real_train = 'clean_data/train/real'
real_test = 'clean_data/test/real'
ai_train = 'clean_data/train/ai'
ai_test = 'clean_data/test/ai'
X_train, X_test, Y_train, Y_test = [], [], [], []

for img_file in os.listdir(real_train):
    X_train.append(extract_hog_features(os.path.join(real_train, img_file)))
    Y_train.append(0)  # 標記為 real (0)
for img_file in os.listdir(real_test):
    X_test.append(extract_hog_features(os.path.join(real_test, img_file)))
    Y_test.append(0)
for img_file in os.listdir(ai_train):
    X_train.append(extract_hog_features(os.path.join(ai_train, img_file)))
    Y_train.append(1)  # 標記為 AI (1)
for img_file in os.listdir(ai_test):
    X_test.append(extract_hog_features(os.path.join(ai_test, img_file)))
    Y_test.append(1)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# 設定隨機種子，確保可復現
random_seed = 42

# Shuffle X_train 和 Y_train
X_train, Y_train = shuffle(X_train, Y_train, random_state=random_seed)

# 訓練 SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, Y_train)

# 預測 & 評估
y_pred = svm.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
report = classification_report(Y_test, y_pred)
roc_auc = roc_auc_score(Y_test, svm.predict_proba(X_test)[:, 1])

# 繪製 Confusion Matrix
plt.figure(figsize=(9,9))
sns.heatmap(conf_matrix, annot=True, fmt=".3f", linewidths=0.5, square=True, cmap="mako")
plt.ylabel('Actual Label', size=12)
plt.xlabel('Predicted Label', size=12)
plt.title(f'Accuracy Score: {accuracy:.4f}', size = 16)
plt.show()

# 交叉驗證
cv_scores = cross_val_score(svm, X_train, Y_train, cv=5)

# 繪製 ROC 曲線
fpr, tpr, _ = roc_curve(Y_test, svm.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'AUROC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 輸出結果
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'AUROC = {roc_auc:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{report}')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')