import pandas as pd
from pathlib import Path
from enum import StrEnum

class DataType(StrEnum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    RETURNS = "returns"

class Data:
    def __init__(self, path: Path = Path(__file__).parent / "data"):
        self.path = Path(path)
        
        self.data = {
            field: pd.read_csv(self.path / f"{field.value}.csv", index_col="Date", parse_dates=True)
            for field in DataType if field !=  DataType.RETURNS
        }
        self.data[DataType.RETURNS] = self.data[DataType.CLOSE].pct_change(1, fill_method=None)

    @property
    def open(self) -> pd.DataFrame:
        return self.data[DataType.OPEN]

    @property
    def high(self) -> pd.DataFrame:
        return self.data[DataType.HIGH]

    @property
    def low(self) -> pd.DataFrame:
        return self.data[DataType.LOW]

    @property
    def close(self) -> pd.DataFrame:
        return self.data[DataType.CLOSE]

    @property
    def volume(self) -> pd.DataFrame:
        return self.data[DataType.VOLUME]

    @property
    def returns(self) -> pd.DataFrame:
        return self.data[DataType.RETURNS]

    def __call__(self, field: DataType) -> pd.DataFrame:
        """возрвашает даннные:
        field: тип данных
        return:
            данные в формате pandas
        """
        return self.data[field]
