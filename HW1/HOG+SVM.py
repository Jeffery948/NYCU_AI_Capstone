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
from sklearn.decomposition import PCA

# Flip image horizontally
def flip_image(image):
    return cv2.flip(image, 1)

# Rotate image by small angle
def rotate_image(image, angle=10):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Adjust brightness and contrast
def adjust_brightness_contrast(image, alpha=1.2, beta=10):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Add Gaussian noise
def add_gaussian_noise(image, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

# Set HOG parameters
hog_params = dict(
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    orientations=9,
    block_norm='L2-Hys'
)

def extract_hog_features(image_path, augment=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    image = cv2.resize(image, (512, 512))  # Resize image
    if augment:
        image = add_gaussian_noise(image)
    features = hog(image, **hog_params)
    return features

# Read data
real_train = 'clean_data/train/real'
real_test = 'clean_data/test/real'
ai_train = 'clean_data/train/ai'
ai_test = 'clean_data/test/ai'
X_train, X_test, Y_train, Y_test = [], [], [], []

for img_file in os.listdir(real_train):
    X_train.append(extract_hog_features(os.path.join(real_train, img_file)))
    #X_train.append(extract_hog_features(os.path.join(real_train, img_file), augment=True))
    Y_train.append(0)  # Label as real (0)
    #Y_train.append(0)
for img_file in os.listdir(real_test):
    X_test.append(extract_hog_features(os.path.join(real_test, img_file)))
    Y_test.append(0)
for img_file in os.listdir(ai_train):
    X_train.append(extract_hog_features(os.path.join(ai_train, img_file)))
    #X_train.append(extract_hog_features(os.path.join(ai_train, img_file), augment=True))
    Y_train.append(1)  # Label as AI (1)
    #Y_train.append(1)
for img_file in os.listdir(ai_test):
    X_test.append(extract_hog_features(os.path.join(ai_test, img_file)))
    Y_test.append(1)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Set random seed for reproducibility
random_seed = 42

# Shuffle X_train and Y_train
X_train, Y_train = shuffle(X_train, Y_train, random_state=random_seed)
print(len(X_train), len(Y_train), len(X_test), len(Y_test), X_train.shape, X_test.shape)

'''pca = PCA()  # Calculate how many components can retain 95% variance after dimensionality reduction
pca.fit(X_train)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to retain 95% variance: {n_components_95}")
exit(0)'''

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape, X_test_pca.shape)

# Train SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_pca, Y_train)

# Prediction & Evaluation
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
report = classification_report(Y_test, y_pred)
roc_auc = roc_auc_score(Y_test, svm.predict_proba(X_test_pca)[:, 1])

# Plot Confusion Matrix
plt.figure(figsize=(9, 9))
sns.heatmap(conf_matrix, annot=True, fmt=".3f", linewidths=0.5, square=True, cmap="mako")
plt.ylabel('Actual Label', size=12)
plt.xlabel('Predicted Label', size=12)
plt.title(f'Accuracy Score: {accuracy:.4f}', size=16)
plt.savefig('SVM_graph/confusion_matrix_SVM13.jpg')
plt.show()

# Cross-validation
cv_scores = cross_val_score(svm, X_train_pca, Y_train, cv=5)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(Y_test, svm.predict_proba(X_test_pca)[:, 1])
plt.plot(fpr, tpr, label=f'AUROC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('SVM_graph/ROC_SVM13.jpg')
plt.show()

# Output results
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'AUROC = {roc_auc:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{report}')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')