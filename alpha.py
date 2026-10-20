import numpy as np
import pandas as pd
from collections.abc import Callable
from data import Data,DataType




class Alpha:

    data = Data()
    
    def __init__(self,kernel:Callable):
        self.kernel = kernel
        self.alpha = None


    def _rank(self,signal:pd.DataFrame)-> pd.DataFrame:
        return signal.rank(axis=1,pct=True) - 0.5

    def _netralization(self,signal:pd.DataFrame)->pd.DataFrame:
        pos = np.clip(signal,0,None)
        neg = np.clip(signal,None,0)
        sum_pos = pos.sum(axis=1)
        sum_neg = neg.abs().sum(axis=1)
        w_pos = pos.div(sum_pos,axis = 0).fillna(0)*0.5
        w_neg = neg.div(sum_neg,axis = 0).fillna(0)*0.5
        return w_pos + w_neg

    def pnl(self)->pd.Series:
        if self.alpha is None:
            self.__call__()
        pnl = (self.data(DataType.CLOSE).pct_change(1).shift(-1)*
        self.alpha).sum(axis=1)
        
        return pnl

    def  sharpe_ratio(self)->float:
        pnl = self.pnl()
        return (pnl.mean()/pnl.std())*np.sqrt(252)

    def turnover(self)->pd.Series:
        if self.alpha is None:
            self.__call__()
        return self.alpha.diff().abs().sum(axis=1)

    def __call__(self)->pd.DataFrame:
        """запуск альфы"""
        signal = self.kernel(self.data)
        signal = self._rank(signal)
        self.alpha = self._netralization(signal)
        return self.alpha
        
        

        
        

    
