from sklearn.decomposition import PCA
import numpy as np

class SoftPCA(PCA):
    """
    Drop-in replacement for sklearn.decomposition.PCA that implements 
    'Soft' Whitening with a tunable epsilon.
    
    Whitening formula:
        X_new = X_projected / sqrt(explained_variance + epsilon)
    
    Args:
        n_components (int/float/str): Number of components to keep.
        epsilon (float): Regularization parameter. 
                         Higher = Softer whitening (preserves more variance info).
                         Lower = Harder whitening (forces spherical distribution).
        **kwargs: Passed to sklearn.decomposition.PCA (e.g., svd_solver, random_state).
    """
    def __init__(self, n_components=None, epsilon=None, **kwargs):
        # Force whiten=False in parent because we handle whitening manually
        super().__init__(n_components=n_components, whiten=False, **kwargs)
        self.epsilon = epsilon

    def transform(self, X):
        # 1. Project data using standard PCA (Rotate to principal axes)
        X_transformed = super().transform(X)
        
        if self.epsilon is not None:
            # 2. Apply Soft Whitening (Scale by inverse sqrt of eigenvalues + epsilon)
            # self.explained_variance_ contains the eigenvalues (lambda)
            scale = np.sqrt(self.explained_variance_ + self.epsilon)
            
            # Avoid division by zero (unlikely with epsilon > 0, but good practice)
            scale[scale == 0] = 1e-10
            
            return X_transformed / scale
        else:
            # No whitening, just return PCA projection
            return X_transformed

    def fit_transform(self, X, y=None):
        # Standard fit, then our custom transform
        self.fit(X, y)
        return self.transform(X)