import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from utils import load_dataset
from pca import *
from ann import ANN
 
DATASET_PATH = "dataset/faces"  
IMG_SIZE = (50, 50)
K_VALUES = [5, 10, 20, 30, 50, 80]   
IMPOSTER_THRESHOLD = 50.0              
HIDDEN_SIZE = 128
EPOCHS = 1000
LR = 0.01
 
 
print("Loading dataset...")
face_db, labels, label_names = load_dataset(DATASET_PATH, IMG_SIZE)
n_classes = len(label_names)
print(f"  Total images: {face_db.shape[1]}, Classes: {n_classes}")

 
X_train, X_test, y_train, y_test = train_test_split(
    face_db.T, labels, test_size=0.4, stratify=labels, random_state=42
)
X_train, X_test = X_train.T, X_test.T   

 
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
 
    Omega_train = compute_signatures(eigenfaces, phi_train)   
    Omega_test  = eigenfaces @ phi_test                          
 
    Y_train_oh = label_binarize(y_train, classes=range(n_classes)).T   
 
    ann = ANN(input_size=k, hidden_size=HIDDEN_SIZE, output_size=n_classes, lr=LR)
    ann.train(Omega_train, Y_train_oh, epochs=EPOCHS)
 
    preds = ann.predict(Omega_test)
    acc = np.mean(preds == y_test) * 100
    accuracies.append(acc)
    print(f"  Accuracy at k={k}: {acc:.2f}%")

 
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