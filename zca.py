import numpy as np

class ZCAWhitening:
    """
    Soft-ZCA Whitening implementation based on 'Isotropy Matters' (arXiv:2411.17538v2).
    
    Formula: W = U * (Lambda + epsilon * I)^(-1/2) * U.T
    where U, Lambda come from SVD of the covariance matrix.
    
    Attributes:
        epsilon (float): Regularization parameter. 
                         Paper suggests values like 0.1 or 0.01 for Soft-ZCA.
                         Use very small values (e.g., 1e-5) for standard ZCA behavior.
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.mean_ = None
        self.W_ = None

    def fit(self, X):
        """
        Fit the ZCA whitening matrix on the training data.
        X: numpy array of shape [n_samples, n_features]
        """
        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Compute Covariance Matrix
        # rowvar=False assumes X is [samples, features]
        Sigma = np.cov(X_centered, rowvar=False) 
        
        # 3. SVD Decomposition
        # U: eigenvectors [features, features]
        # Lambda: eigenvalues [features]
        U, Lambda, _ = np.linalg.svd(Sigma)
        
        # 4. Compute Whitening Matrix W
        # Using the Soft-ZCA regularization formula
        inv_sqrt_lambda = np.diag(1.0 / np.sqrt(Lambda + self.epsilon))
        self.W_ = np.dot(U, np.dot(inv_sqrt_lambda, U.T))
        
        return self

    def transform(self, X):
        """
        Apply the learned whitening transformation to X.
        """
        if self.mean_ is None or self.W_ is None:
            raise RuntimeError("ZCAWhitening has not been fitted yet.")
            
        X_centered = X - self.mean_
        return np.dot(X_centered, self.W_.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)