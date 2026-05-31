import os
import cv2
import numpy as np

def load_dataset(dataset_path, img_size=(50, 50)):
    """
    Loads all face images from dataset directory.
    Returns face_db (mn x p), labels, and label_names.
    Dataset folder structure: dataset/person_name/img1.jpg ...
    """
    face_db = []
    labels = []
    label_names = []

    for label_idx, person in enumerate(sorted(os.listdir(dataset_path))):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        label_names.append(person)
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)        
            col_vec = img.flatten().astype(np.float64)  
            face_db.append(col_vec)
            labels.append(label_idx)

    face_db = np.array(face_db).T   
    labels = np.array(labels)
    return face_db, labels, label_names