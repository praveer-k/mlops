import pandas as pd
from pydantic import ValidationError
from typing import List, Tuple

from mlops.schema import CustomerFeatures

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df

    def validate(self) -> Tuple[List[CustomerFeatures], List[Tuple[int, str]]]:
        valid_records = []
        errors = []

        for idx, row in self.df.iterrows():
            try:
                record = CustomerFeatures(**row.to_dict())
                valid_records.append(record)
            except ValidationError as e:
                errors.append((idx, e.errors()))

        if len(errors) > 0:
            raise Exception("Data validation failed for a few rows")
        return valid_records, errors
