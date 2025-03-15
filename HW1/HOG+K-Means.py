import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

# Left-right flip
def flip_image(image):
    return cv2.flip(image, 1)

# Small angle rotation
def rotate_image(image, angle=10):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Adjust brightness and contrast
def adjust_brightness_contrast(image, alpha=1.2, beta=10):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Add noise
def add_gaussian_noise(image, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

# Set HOG parameters
hog_params = dict(
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    orientations=9,
    block_norm='L2-Hys'
)

def extract_hog_features(image_path, augment=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    image = cv2.resize(image, (512, 512))  # Resize
    if augment:
        image = add_gaussian_noise(image)
    features = hog(image, **hog_params)
    return features

# Load data
real_train = 'clean_data/train/real'
real_test = 'clean_data/test/real'
ai_train = 'clean_data/train/ai'
ai_test = 'clean_data/test/ai'
X_full, y_true = [], []
for img_file in os.listdir(real_train):
    X_full.append(extract_hog_features(os.path.join(real_train, img_file)))
    #X_full.append(extract_hog_features(os.path.join(real_train, img_file), augment=True))
    y_true.append(0)  # Label as real (0)
    #y_true.append(0)
for img_file in os.listdir(real_test):
    X_full.append(extract_hog_features(os.path.join(real_test, img_file)))
    #X_full.append(extract_hog_features(os.path.join(real_test, img_file), augment=True))
    y_true.append(0)
    #y_true.append(0)
for img_file in os.listdir(ai_train):
    X_full.append(extract_hog_features(os.path.join(ai_train, img_file)))
    #X_full.append(extract_hog_features(os.path.join(ai_train, img_file), augment=True))
    y_true.append(1)  # Label as AI (1)
    #y_true.append(1)
for img_file in os.listdir(ai_test):
    X_full.append(extract_hog_features(os.path.join(ai_test, img_file)))
    #X_full.append(extract_hog_features(os.path.join(ai_test, img_file), augment=True))
    y_true.append(1)
    #y_true.append(1)

X_full = np.array(X_full)
y_true = np.array(y_true)

# Set random seed for reproducibility
random_seed = 42

# Shuffle X_train and Y_train
X_full, y_true = shuffle(X_full, y_true, random_state=random_seed)
print(len(X_full), len(y_true), X_full.shape, y_true.shape)

pca = PCA(n_components=10)
X_full = pca.fit_transform(X_full)
print(X_full.shape)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=random_seed)
y_pred = kmeans.fit_predict(X_full)

# Evaluation
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)

# Compute Clustering Accuracy (Hungarian Algorithm)
contingency_matrix = np.zeros((2, 2))
for i in range(len(y_true)):
    contingency_matrix[y_true[i], y_pred[i]] += 1

row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
accuracy = contingency_matrix[row_ind, col_ind].sum() / len(y_true)

# Remap clusters to match true labels
mapped_clusters = np.zeros_like(y_pred)
for i, j in zip(row_ind, col_ind):
    mapped_clusters[y_pred == j] = i

# Reduce dimensions for visualization
X_embedded = TSNE(n_components=2, random_state=random_seed, perplexity=30).fit_transform(X_full)

colors = ['blue', 'red']
label_names = ["Real", "AI"]

# Plot Clusters
plt.figure(figsize=(6, 5))
for i, label in enumerate(np.unique(y_true)):  # Iterate over unique labels (Only 0 and 1)
    plt.scatter(
        X_embedded[y_true == label, 0], 
        X_embedded[y_true == label, 1], 
        c=colors[i], 
        label=label_names[i], 
        alpha=0.6
    )
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='X', s=100, label="Centers")
plt.title('K-Means Clustering')
plt.legend()
plt.savefig('K_graph/result17.jpg')
plt.show()

# Output results
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")
print(f"Clustering accuracy: {accuracy * 100:.2f}")