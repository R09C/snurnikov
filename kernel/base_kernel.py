from abc import ABC, abstractmethod
import pandas as pd
from ..data import Data


class BaseKernel(ABC):
    @abstractmethod
    def __call__(self, data: Data) -> pd.DataFrame:
        pass
    

    