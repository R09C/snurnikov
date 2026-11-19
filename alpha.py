import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Callable
from data import Data,DataType




class Alpha():

    data = Data()
    
    def __init__(self,kernel:Callable,delay:int = 1,rank:bool = False,
                 cut_outliers:int = 0,cut_middle:int = 0,
                 truncate:float = None,decay:int = 0):
        self.kernel = kernel
        self.delay = delay
        self.rank = rank
        self.cut_outliers = cut_outliers
        self.cut_middle = cut_middle
        self.truncate = truncate
        self.decay = decay
        self.position = None


    def _rank(self,signal:pd.DataFrame)-> pd.DataFrame:
        r = signal.rank(axis=1)
        n = signal.count(axis=1)
        return r.sub(1,axis=0).div(n-1,axis=0)

    def _netralization(self,signal:pd.DataFrame)->pd.DataFrame:
        return signal.sub(signal.mean(axis=1),axis=0)
    
    def _normalization(self,signal:pd.DataFrame)->pd.DataFrame:
        gross = signal.abs().sum(axis=1)
        return signal.div(gross,axis = 0).replace([np.inf,-np.inf],np.nan)

    def _truncate(self,signal:pd.DataFrame,threshold:float)->pd.DataFrame:
        return signal.clip(-threshold,threshold)

    def _cut_outliers(self,signal:pd.DataFrame,n:int)->pd.DataFrame:
        order = signal.rank(axis=1,method='first')
        cnt = signal.count(axis=1)
        mask = order.le(n) | order.gt(cnt-n,axis=0)
        return signal.mask(mask,0)

    def _cut_middle(self,signal:pd.DataFrame,n:int)->pd.DataFrame:
        order = signal.rank(axis=1,method='first')
        cnt = signal.count(axis=1)
        start = (cnt-n)//2 + 1
        mask = order.ge(start,axis=0) & order.lt(start+n,axis=0)
        return signal.mask(mask,0)

    def _decay(self,position:pd.DataFrame,n:int)->pd.DataFrame:
        w = np.arange(n,0,-1)
        decayed = sum(w[k]*position.shift(k,fill_value=0) for k in range(n))/w.sum()
        return self._normalization(decayed).fillna(0)
    

    def pnl(self)->pd.Series:
        if self.position is None:
            self.__call__()
        pnl = (self.data.returns *
        self.position.shift(self.delay)).sum(axis=1)
        
        return pnl

    def  sharpe_ratio(self,pnl:pd.Series = None)->float:
        if pnl is None:
            pnl = self.pnl()
        return (pnl.mean()/pnl.std())*np.sqrt(252)

    def turnover(self)->pd.Series:
        if self.position is None:
            self.__call__()
        return self.position.diff().abs().sum(axis=1)

    def cumulative(self)->pd.Series:
        return self.pnl().cumsum()

    def max_drawdown(self,cum:pd.Series = None)->float:
        if cum is None:
            cum = self.cumulative()
        return (cum.cummax().clip(lower=0) - cum).max()

    def correlation(self,other)->float:
        return self.pnl().corr(other.pnl())

    def alpha_stats(self)->pd.DataFrame:
        pnl = self.pnl()
        turnover = self.turnover()
        stats = {}
        for year,p in pnl.groupby(pnl.index.year):
            stats[year] = {
                "sharpe":self.sharpe_ratio(p),
                "turnover":turnover.loc[p.index].mean(),
                "return":p.sum(),
                "drawdown":self.max_drawdown(p.cumsum()),
            }
        pnl.cumsum().plot(xlabel="Дата",ylabel="Накопленная доходность")
        return pd.DataFrame(stats).T

    def check(self,others:list = None)->dict:
        pnl = self.pnl()
        yearly_sharpe = pnl.groupby(pnl.index.year).apply(self.sharpe_ratio)
        cum_ret = pnl.sum()
        mean_to = self.turnover().mean()
        good_years = (yearly_sharpe >= 1).sum()
        max_corr = 0
        if others:
            max_corr = max(abs(self.correlation(o)) for o in others)
        return {
            "return":round(cum_ret,3),
            "turnover":round(mean_to,3),
            "years_sharpe>=1":int(good_years),
            "max_corr":round(float(max_corr),3),
            "passed":bool(cum_ret >= 0.4 and mean_to <= 0.8 and good_years >= 4 and max_corr <= 0.5),
        }

    def volatility(self)->pd.Series:
        vol = self.data.returns.std().dropna().sort_values()
        vol.reset_index(drop=True).plot(xlabel="Инструмент",ylabel="Волатильность")
        return vol

    def __call__(self)->pd.DataFrame:
        """запуск альфы"""
        signal = self.kernel(self.data).replace([np.inf,-np.inf],np.nan)
        if self.rank:
            signal = self._rank(signal)
        signal = self._netralization(signal)
        if self.cut_outliers:
            signal = self._cut_outliers(signal,self.cut_outliers)
        if self.cut_middle:
            signal = self._cut_middle(signal,self.cut_middle)
        pos = self._normalization(self._netralization(signal))
        if self.truncate is not None:
            pos = self._normalization(self._netralization(self._truncate(pos,self.truncate)))
        pos = pos.fillna(0)
        if self.decay > 1:
            pos = self._decay(pos,self.decay)
        self.position = pos
        return self.position
