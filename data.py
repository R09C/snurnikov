import pandas as pd
from pathlib import Path
from enum import StrEnum

class DataType(StrEnum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    TICKVOLUME = "tickvolume"

class Data:
    def __init__(self, path: Path = Path(r"D:\sirius\alg_stat_T\snurnikov\data")):
        self.path = Path(path)
        
        self.data = {
            field: pd.read_csv(self.path / f"{field.value}.csv", index_col="Date", parse_dates=True)
            for field in DataType
        }
    
    def __call__(self, field: DataType) -> pd.DataFrame:
        """возрвашает даннные:
        field: тип данных
        return:
            данные в формате pandas
        """
        return self.data[field]
