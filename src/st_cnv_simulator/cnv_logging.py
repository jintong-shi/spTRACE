# st_cnv_simulator/cnv_logging.py
from typing import Any, Dict, List
import pandas as pd

class EventLogger:
    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []

    def log(self, row: Dict[str, Any]) -> None:
        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=[
                "iter","cell_id","parent_id","clone_id","cell_type",
                "source","row_id","Chromosome","Start","End",
                "Major_Copy_Number","Minor_Copy_Number","length_mb",
                "hap_assigned","affected_genes"
            ])
        df = pd.DataFrame(self._rows)
        if "affected_genes" in df.columns:
            df["affected_genes"] = df["affected_genes"].apply(
                lambda x: ";".join(map(str, x)) if isinstance(x, (list, tuple)) else x
            )
        return df
