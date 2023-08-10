from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
class DataFrameImputerUtils(TransformerMixin):
    def __int__(self):
        pass
    def fit(self, X, y=None):
        # 如果类型是object就统计个数 如果不是的话，就求平均
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

