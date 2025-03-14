import matplotlib.pyplot as plt
import numpy as np

# Data (updated as per the table provided)
label = ["311,364", "450", "400", "200", "100", "50", "10"]
data_quantity = np.arange(len(label))
accuracy = np.array([77.33, 77.33, 72.67, 76.67, 78.00, 80.00, 78.00])
auroc = np.array([0.86, 0.86, 0.82, 0.80, 0.85, 0.86, 0.86])
cv_accuracy = np.array([0.7600, 0.7600, 0.6644, 0.6711, 0.7244, 0.7711, 0.8356])
cv_std = np.array([0.0565, 0.0565, 0.0661, 0.0389, 0.0606, 0.0480, 0.0653])

# Plot
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel('Data Dimensions')
ax1.set_ylabel('Accuracy (%)', color='tab:blue')
ax1.plot(data_quantity, accuracy, 'o-', color='tab:blue', label='Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticklabels(label, rotation=20, ha="right")

ax2 = ax1.twinx()
ax2.set_ylabel('AUROC / CV Accuracy', color='tab:red')
ax2.plot(data_quantity, auroc, 's-', color='tab:red', label='AUROC')
ax2.errorbar(data_quantity, cv_accuracy, yerr=cv_std, fmt='d-', color='tab:green', label='CV Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Model Performance vs. Data Dimensions')
plt.show()
