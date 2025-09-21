# Real vs. AI-Generated Animal Photos Recognition

This project aims to distinguish between **real animal photos** and **AI-generated images** using classical machine learning methods. The experiments compare **supervised learning** (SVM, Logistic Regression) and **unsupervised learning** (K-Means) approaches, and further investigate the impact of **data quantity, data augmentation, and dimensionality reduction**.

---

## 📂 Dataset

- **Real images**: 300 animal photos collected from [Unsplash](https://unsplash.com/s/photos/animal?license=free) using a web crawler.  
- **AI images**: 300 images generated with **Stable Diffusion Web UI** using 300 prompts provided by ChatGPT.  

### Preprocessing
- Training set: **225 real + 225 AI images**  
- Testing set: **75 real + 75 AI images**  
- Each image is resized to **512×512** and converted to grayscale before feature extraction.  

### Dataset Structure
```
HW1/
├── raw_data/
│   ├── real_animal_images/
│   └── ai_animal_images/
└── clean_data/
    ├── train/
    │   ├── real/
    │   └── ai/
    └── test/
        ├── real/
        └── ai/
```

---

## ⚙️ Experimental Setup

### Feature Extraction
- **Histogram of Oriented Gradients (HOG)** used to extract features.
- Dimensionality reduction with **PCA** explored (retaining 95% variance → ~399 components, but further reduced to 10–450 in experiments).

### Methods
1. **Support Vector Machine (SVM)**
   - Kernels: linear, polynomial, sigmoid
   - Evaluation metrics: Accuracy, Confusion Matrix, Precision, Recall, F1, AUROC, Cross-validation

2. **Logistic Regression**
   - Penalties: L1, L2
   - Evaluation metrics: Same as SVM

3. **K-Means Clustering**
   - Unsupervised baseline
   - Evaluation metrics: Accuracy (via Hungarian Algorithm), Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
   - Visualization with **t-SNE**

---

## 🧪 Experiments and Results

### 1. Support Vector Machine (SVM)
- **Best baseline**:  
  - HOG (cells per block = (3,3)), resize = 512×512, kernel = linear  
  - Accuracy = **77.33%**, AUROC = **0.86**  

- **Findings**:  
  - More training data generally improves performance (best at 300 images).  
  - Data augmentation improves cross-validation (esp. flipping), but Gaussian noise hurts.  
  - Dimensionality reduction improves performance (best = 80% accuracy with 50 components).

---

### 2. Logistic Regression
- **Best baseline**:  
  - HOG (cells per block = (3,3)), resize = 512×512, **L1 penalty**  
  - Accuracy = **70%**, AUROC = **0.76**  

- **Findings**:  
  - L1 performs better than L2 due to sparsity in high-dimensional features.  
  - Accuracy improves with more training data, but not strictly monotonic.  
  - Data augmentation has mixed effects; Gaussian noise degrades results.  
  - Dimensionality reduction to 50–10 components boosts accuracy to **77.33%**.

---

### 3. K-Means Clustering
- **Best baseline**:  
  - HOG (cells per block = (2,2)), resize = 512×512  
  - Accuracy = **64%**, ARI = **0.077**, NMI = **0.067**  

- **Findings**:  
  - Performance improves with more data (peaks at 300 images).  
  - Data augmentation worsens clustering (features become inconsistent).  
  - Dimensionality reduction hurts performance at 200–400 components, but recovers at ~100–10.  

---

## 📊 Discussion

- **SVM** outperforms Logistic Regression and K-Means, with **linear kernel** proving most effective.  
- **Data augmentation** benefits supervised learning but not clustering.  
- **Dimensionality reduction (PCA)** helps both supervised models, showing that many HOG features are redundant.  
- **Unsupervised clustering** is less reliable for this task but provides insights into feature space structure.  

---

## 🔮 Future Work

- Broader **hyperparameter optimization** (e.g., SVM C values, Logistic Regression solvers).  
- More advanced **augmentation** (zoom, crop, motion blur).  
- Explore **ensemble methods** or **deep learning models (CNNs)** for feature extraction and classification.  
- Compare **other dimensionality reduction techniques** (t-SNE, UMAP) with PCA.  

---

## 📌 References

- [Unsplash Animals Dataset](https://unsplash.com/s/photos/animal?license=free)  
- [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)  
- [scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)  
- [scikit-image HOG](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)  
