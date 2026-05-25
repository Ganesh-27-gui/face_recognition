import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from utils import load_dataset
from pca import *
from ann import ANN

# ── CONFIG ──────────────────────────────────────────────────────
DATASET_PATH = "dataset/faces"  # path to dataset folder
IMG_SIZE = (50, 50)
K_VALUES = [5, 10, 20, 30, 50, 80]   # values of k to test
IMPOSTER_THRESHOLD = 50.0             # Euclidean distance threshold for imposters
HIDDEN_SIZE = 128
EPOCHS = 1000
LR = 0.01
# ────────────────────────────────────────────────────────────────

# Load data
print("Loading dataset...")
face_db, labels, label_names = load_dataset(DATASET_PATH, IMG_SIZE)
n_classes = len(label_names)
print(f"  Total images: {face_db.shape[1]}, Classes: {n_classes}")

# Train/test split (60/40)
X_train, X_test, y_train, y_test = train_test_split(
    face_db.T, labels, test_size=0.4, stratify=labels, random_state=42
)
X_train, X_test = X_train.T, X_test.T   # back to (mn, samples)

# PCA Steps
mean_face = compute_mean(X_train)
phi_train = mean_zero(X_train, mean_face)
phi_test  = mean_zero(X_test,  mean_face)
C = compute_surrogate_covariance(phi_train)
eigenvalues, eigenvectors = get_eigenvectors(C)

accuracies = []

for k in K_VALUES:
    print(f"\n── k = {k} ──")
    V_k = get_feature_vectors(eigenvectors, k)
    eigenfaces = compute_eigenfaces(V_k, phi_train)
    
    # Signatures (projected features)
    Omega_train = compute_signatures(eigenfaces, phi_train)  # (k, train_samples)
    Omega_test  = eigenfaces @ phi_test                       # (k, test_samples)

    # One-hot encode labels
    Y_train_oh = label_binarize(y_train, classes=range(n_classes)).T  # (classes, samples)

    # Build and train ANN
    ann = ANN(input_size=k, hidden_size=HIDDEN_SIZE, output_size=n_classes, lr=LR)
    ann.train(Omega_train, Y_train_oh, epochs=EPOCHS)

    # Evaluate
    preds = ann.predict(Omega_test)
    acc = np.mean(preds == y_test) * 100
    accuracies.append(acc)
    print(f"  Accuracy at k={k}: {acc:.2f}%")

# Plot Accuracy vs k
plt.figure(figsize=(8, 5))
plt.plot(K_VALUES, accuracies, marker='o', linewidth=2, color='steelblue')
plt.title("Accuracy vs Number of Eigenfaces (k)")
plt.xlabel("k (Number of Eigenfaces)")
plt.ylabel("Accuracy (%)")
plt.xticks(K_VALUES)
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=150)
plt.show()
print("\nPlot saved as accuracy_vs_k.png")