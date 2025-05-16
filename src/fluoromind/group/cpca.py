import numpy as np
from typing import Optional, List, Any

from ..algorithms import CPCA, CPCAResult


class GroupCPCA(CPCA):
    """Group-level Complex Principal Component Analysis (CPCA).

    A variant of CPCA that handles group-level data by concatenating subjects along the time dimension.
    Inherits from CPCA and adds functionality to extract subject-specific results.
    """

    def __init__(self, n_components: Optional[int] = None, threshold: float = 1.0, random_state: Optional[int] = None):  # CPCA 基类不接受‘n_patterns’作为参数，使用了 n_components
        self.threshold = threshold
        self.random_state = random_state
        super().__init__(n_components=n_components)
        self.result_ = None
        self.n_samples_: Optional[List[int]] = None
        self.start_indices_: Optional[np.ndarray[Any, np.dtype[Any]]] = None
        self.subject_explained_variance_: Optional[List[np.ndarray]] = None

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
        params = super().get_params(deep=deep)
        params['threshold'] = self.threshold
        params['random_state'] = self.random_state
        return params  #

    def set_params(self, **params) -> "GroupCPCA":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : GroupCPCA
            Estimator instance.
        """
        super().set_params(**params)
        return self
        self.subject_explained_variance_ratio: Optional[List[np.ndarray]] = None

    def fit(self, data: List[np.ndarray]) -> "GroupCPCA":
        self.n_samples_ = [sub_data.shape[0] for sub_data in data]
        self.start_indices_ = np.insert(np.cumsum(self.n_samples_), 0, 0)

        data_2d = np.concatenate(data, axis=0)
        self.result_ = super().fit(data_2d)  # 结果存储在 self.result_（实例属性）中，使得 fit() 计算的结果可以在 整个对象的生命周期内被访问。

        # Calculate subject-specific explained variance
        self.subject_explained_variance_ = []
        self.subject_explained_variance_ratio_ = []

        for i in range(len(self.n_samples_)):
            start_idx = self.start_indices_[i]
            end_idx = self.start_indices_[i + 1]
            sub_data = data_2d[start_idx:end_idx]
            sub_scores = self.result_.scores_[start_idx:end_idx]  # scores_ 代表 模型训练后计算出的属性（以 _ 结尾），符合 Scikit-Learn 规范，适用于模型训练后的属性。

            # Calculate subject-specific variances
            sub_var = np.var(sub_data, ddof=1, axis=0).sum()
            explained_var = np.var(sub_scores, ddof=1, axis=0)
            explained_var_ratio = explained_var / sub_var

            self.subject_explained_variance_.append(explained_var)
            self.subject_explained_variance_ratio_.append(explained_var_ratio)

        return self  # fit() 方法应该返回 self，以支持方法链式调用

    def __len__(self) -> int:
        if self.n_samples_ is None:
            raise ValueError("GroupCPCA model not fitted yet.")
        return len(self.n_samples_)

    def __getitem__(self, index: int) -> CPCAResult:
        if self.result_ is None:
            raise ValueError("GroupCPCA model not fitted yet.")

        start_idx = self.start_indices_[index]
        end_idx = self.start_indices_[index + 1]
        return CPCAResult(
            components=self.result_.components_,  # self.result_.components_ 是 GroupCPCA 通过 fit() 方法计算得到的主成分。这个值是在模型拟合（训练）之后得到的，因此它带有下划线。
            scores=self.result_.scores_[start_idx:end_idx],   # scores_
            singular_values=self.result_.singular_values_,
            explained_variance=self.subject_explained_variance_[index],
            explained_variance_ratio=self.subject_explained_variance_ratio_[index],
            n_components=self.result_.n_components,
            force_complex=self.result_.force_complex,
            standardize=self.result_.standardize,
        )
