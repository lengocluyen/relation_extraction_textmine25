from glob import glob
import os

import pandas as pd


def load_data(
    dir_or_file_path: str, index_col=None, sep=",", ignore_index=False
) -> pd.DataFrame:
    if os.path.isdir(dir_or_file_path):
        all_files = glob(os.path.join(dir_or_file_path, "*.csv")) + glob(
            os.path.join(dir_or_file_path, "*.parquet")
        )
    else:
        assert dir_or_file_path.endswith(".csv") or dir_or_file_path.endswith(
            ".parquet"
        )
        all_files = [dir_or_file_path]
    assert len(all_files) > 0
    return pd.concat(
        [
            (
                pd.read_csv(filename, index_col=index_col, header=0, sep=sep)
                if filename.endswith(".csv")
                else pd.read_parquet(filename)
            )
            for filename in all_files
        ],
        axis=0,
        ignore_index=ignore_index,
    )
