import numpy as np

def compute_mean(face_db):
    """Step 2: Mean vector (mn, 1)"""
    return np.mean(face_db, axis=1, keepdims=True)  # shape: (mn, 1)

def mean_zero(face_db, mean_face):
    """Step 3: Subtract mean from each face"""
    return face_db - mean_face   # broadcasting: (mn, p) - (mn, 1)

def compute_surrogate_covariance(phi):
    """Step 4: Surrogate covariance (p x p) instead of (mn x mn)"""
    p = phi.shape[1]
    C = (phi.T @ phi) / p       # shape: (p, p)
    return C

def get_eigenvectors(C):
    """Step 5: Eigenvalue decomposition"""
    eigenvalues, eigenvectors = np.linalg.eigh(C)  # eigh for symmetric matrix
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

def get_feature_vectors(eigenvectors, k):
    """Step 6: Select top-k eigenvectors → feature matrix (p, k)"""
    return eigenvectors[:, :k]

def compute_eigenfaces(V_k, phi):
    """
    Step 7: Project mean-aligned faces to feature vector space.
    Eigenfaces shape: (k, mn)
    V_k: (p, k), phi: (mn, p)
    """
    eigenfaces = (V_k.T @ phi.T)    # (k, p) @ (p, mn) → (k, mn)
    # Normalize each eigenface
    norms = np.linalg.norm(eigenfaces, axis=1, keepdims=True)
    eigenfaces = eigenfaces / (norms + 1e-10)
    return eigenfaces

def compute_signatures(eigenfaces, phi):
    """
    Step 8: Project each face onto eigenfaces.
    Signatures (Omega) shape: (k, p)
    """
    return eigenfaces @ phi     # (k, mn) @ (mn, p) → (k, p)

def project_test_face(eigenfaces, test_face, mean_face):
    """
    Steps for testing (Steps 1-3):
    - Flatten test face, subtract mean, project onto eigenfaces.
    """
    I1 = test_face.flatten().astype(np.float64).reshape(-1, 1)  # (mn, 1)
    I2 = I1 - mean_face                                          # (mn, 1)
    omega = eigenfaces @ I2                                      # (k, 1)
    return omega