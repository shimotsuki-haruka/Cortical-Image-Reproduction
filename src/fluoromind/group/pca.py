import numpy as np
from typing import Optional, List, Any
from sklearn.decomposition import PCA as SklearnPCA


class GroupPCA:
    """Group-level PCA.

    A variant of PCA that handles group-level data by concatenating subjects along the time dimension.
    Wraps sklearn.decomposition.PCA and adds functionality to extract subject-specific results.
    """

    def __init__(self, n_components: Optional[int] = None, whiten: bool = False, random_state: Optional[int] = None):
        self.pca_ = SklearnPCA(n_components=n_components, whiten=whiten, random_state=random_state)
        self.n_samples_: Optional[List[int]] = None
        self.start_indices_: Optional[np.ndarray[Any, np.dtype[Any]]] = None
        self.subject_explained_variance_: Optional[List[np.ndarray]] = None
        self.subject_explained_variance_ratio_: Optional[List[np.ndarray]] = None
        self.scores_: Optional[np.ndarray] = None

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "n_components": self.pca_.n_components,
            "whiten": self.pca_.whiten,
            "random_state": self.pca_.random_state,
        }

    def set_params(self, **params) -> "GroupPCA":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : GroupPCA
            Estimator instance.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
            setattr(self.pca_, key, value)
        return self

    def fit(self, data: List[np.ndarray]) -> "GroupPCA":
        self.n_samples_ = [sub_data.shape[0] for sub_data in data]
        self.start_indices_ = np.insert(np.cumsum(self.n_samples_), 0, 0)

        data_2d = np.concatenate(data, axis=0)
        self.pca_.fit(data_2d)
        self.scores_ = self.pca_.transform(data_2d)

        # Calculate subject-specific explained variance
        self.subject_explained_variance_ = []
        self.subject_explained_variance_ratio_ = []

        for i in range(len(self.n_samples_)):
            start_idx = self.start_indices_[i]
            end_idx = self.start_indices_[i + 1]
            sub_data = data_2d[start_idx:end_idx]
            sub_scores = self.scores_[start_idx:end_idx]

            # Calculate subject-specific variances
            sub_var = np.var(sub_data, ddof=1, axis=0).sum()
            explained_var = np.var(sub_scores, ddof=1, axis=0)
            explained_var_ratio = explained_var / sub_var

            self.subject_explained_variance_.append(explained_var)
            self.subject_explained_variance_ratio_.append(explained_var_ratio)

        return self

    def __len__(self) -> int:
        if self.n_samples_ is None:
            raise ValueError("GroupPCA model not fitted yet.")
        return len(self.n_samples_)

    def __getitem__(self, index: int) -> dict:
        if not hasattr(self.pca_, "components_"):
            raise ValueError("GroupPCA model not fitted yet.")

        start_idx = self.start_indices_[index]
        end_idx = self.start_indices_[index + 1]
        return {
            "components": self.pca_.components_,  # pca 少了下划线，改为 pca_
            "scores": self.scores_[start_idx:end_idx],
            "singular_values": self.pca_.singular_values_,
            "explained_variance": self.subject_explained_variance_[index],   # 少了下划线
            "explained_variance_ratio": self.subject_explained_variance_ratio_[index],
            "n_components": self.pca_.n_components_,
            "standardize": True if self.pca_.whiten else False,
        }
