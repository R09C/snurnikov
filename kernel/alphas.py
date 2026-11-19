import numpy as np
import pandas as pd
from data import Data
from kernel.base_kernel import BaseKernel
from kernel.reversal import Reversal


class InvClose(BaseKernel):
    def __call__(self, data: Data) -> pd.DataFrame:
        return 1 / data.close

class Atractor(BaseKernel):
    def __init__(self, n=20, vol_q: float = 0.3, vol_win: int = 20):
        self.vol_q = vol_q
        self.vol_win = vol_win
        self.n = n

    def __call__(self, data: Data) -> pd.DataFrame:
        vol = data.returns.rolling(self.vol_win).std()
        mask = vol.gt(vol.quantile(self.vol_q, axis=1), axis=0)
        mean = data.close.rolling(self.n).mean()
        std = data.close.rolling(self.n).std()
        return ((mean - data.close) / std).where(mask)


class HighShadow(BaseKernel):
    def __init__(self, vol_q: float = 0.3, vol_win: int = 20):
        self.vol_q = vol_q
        self.vol_win = vol_win

    def __call__(self, data: Data) -> pd.DataFrame:
        vol = data.returns.rolling(self.vol_win).std()
        mask = vol.lt(vol.quantile(self.vol_q, axis=1), axis=0)
        return ((data.high - np.maximum(data.open, data.close)) / data.close).where(mask)


class UpperWick(BaseKernel):
    def __call__(self, data: Data) -> pd.DataFrame:
        return (data.high - np.maximum(data.open, data.close)) / (data.high - data.low)


class Stoch(BaseKernel):
    def __init__(self, vol_q: float = 0.3, vol_win: int = 20):
        self.vol_q = vol_q
        self.vol_win = vol_win

    def __call__(self, data: Data) -> pd.DataFrame:
        vol = data.returns.rolling(self.vol_win).std()
        mask = vol.lt(vol.quantile(self.vol_q, axis=1), axis=0)
        return (-(data.close - data.low) / (data.high - data.low)).where(mask)


class VolumeRank(BaseKernel):
    def __init__(self, n: int = 35):
        self.n = n

    def __call__(self, data: Data) -> pd.DataFrame:
        return self.ts_rank(data.volume, self.n)


class RSI(BaseKernel):
    def __init__(self, n: int = 14):
        self.n = n

    def __call__(self, data: Data) -> pd.DataFrame:
        return 50 - self.rsi(data.close, self.n)
