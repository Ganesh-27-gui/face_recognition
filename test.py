import numpy as np
import cv2
import os
from utils import load_dataset
from pca import *
from ann import ANN
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import label_binarize 
 
DATASET_PATH    = "dataset/faces"  
IMPOSTER_PATH   = "imposters"   
IMG_SIZE        = (50, 50)
K               = 30            
HIDDEN_SIZE     = 128
EPOCHS          = 1000
LR              = 0.01
CONFIDENCE_THRESHOLD = 0.70  

face_db, labels, label_names = load_dataset(DATASET_PATH, IMG_SIZE)
n_classes = len(label_names)

X_train, X_test, y_train, y_test = train_test_split(
    face_db.T, labels, test_size=0.4, stratify=labels, random_state=42
)
X_train, X_test = X_train.T, X_test.T

# PCA
mean_face   = compute_mean(X_train)
phi_train   = mean_zero(X_train, mean_face)
eigenvalues, eigenvectors = get_eigenvectors(compute_surrogate_covariance(phi_train))
V_k         = get_feature_vectors(eigenvectors, K)
eigenfaces  = compute_eigenfaces(V_k, phi_train)
Omega_train = compute_signatures(eigenfaces, phi_train)
# Train ANN
Y_train_oh = label_binarize(y_train, classes=range(n_classes)).T
ann = ANN(input_size=K, hidden_size=HIDDEN_SIZE, output_size=n_classes, lr=LR)
ann.train(Omega_train, Y_train_oh, epochs=EPOCHS)

def recognize(img_path, ann, eigenfaces, mean_face, label_names, threshold=CONFIDENCE_THRESHOLD):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    omega = project_test_face(eigenfaces, img, mean_face)   
    proba = ann.predict_proba(omega)                        
    max_conf = np.max(proba)
    pred_label = np.argmax(proba)
    
    if max_conf < threshold:
        return "IMPOSTER (Not Enrolled)", max_conf
    else:
        return label_names[pred_label], float(max_conf)
 
print("\n── Known faces test ──")
for i in range(min(5, X_test.shape[1])):
    face_img = X_test[:, i].reshape(IMG_SIZE)
    omega = eigenfaces @ (X_test[:, i:i+1] - mean_face)
    proba = ann.predict_proba(omega)
    conf  = np.max(proba)
    pred  = label_names[np.argmax(proba)]
    actual = label_names[y_test[i]]
    print(f"  Actual: {actual:15s} | Predicted: {pred:15s} | Confidence: {conf:.2f}")

if os.path.exists(IMPOSTER_PATH):
    print("\n── Imposter test ──")
    for img_file in os.listdir(IMPOSTER_PATH):
        img_path = os.path.join(IMPOSTER_PATH, img_file)
        result, conf = recognize(img_path, ann, eigenfaces, mean_face, label_names)
        print(f"  {img_file}: {result} (confidence: {conf:.2f})")
else:
    print(f"\nNo imposter folder found at '{IMPOSTER_PATH}'")