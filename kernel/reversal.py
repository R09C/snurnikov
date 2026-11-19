import pandas as pd
from data import Data
from kernel.base_kernel import BaseKernel


class Reversal(BaseKernel):
    def __init__(self, n: int = 5):
        self.n = n

    def __call__(self, data: Data) -> pd.DataFrame:
        return -data.close / data.close.shift(self.n)
