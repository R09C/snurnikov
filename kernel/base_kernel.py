from abc import ABC, abstractmethod
import pandas as pd
from data import Data


class BaseKernel(ABC):
    def ts_rank(self, x: pd.DataFrame, n: int) -> pd.DataFrame:
        return x.rolling(n).rank(pct=True)

    def ts_corr(self, x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
        return x.rolling(n).corr(y)

    def ts_reg_beta(self, y: pd.DataFrame, x, n: int) -> pd.DataFrame:
        return y.rolling(n).cov(x).div(x.rolling(n).var(), axis=0)

    def rsi(self, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
        diff = close.diff()
        up = diff.clip(lower=0).rolling(n).mean()
        down = (-diff.clip(upper=0)).rolling(n).mean()
        return 100 - 100 / (1 + up / down)

    @abstractmethod
    def __call__(self, data: Data) -> pd.DataFrame:
        pass
