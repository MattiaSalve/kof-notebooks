import enum
import glob
import gc
from pathlib import Path

import pyarrow.parquet as pq
import polars as pl


class Filetypes(enum.StrEnum):
    excel = "excel"
    csv = "csv"


def load_multiple_files(pattern: str, filetype: Filetypes):
    """
    Reads a list of Excel files to a Polars DataFrame
    Args:
        - pattern (str): Glob pattern for the files to match
        - Filetype (StrEnum): csv, excel
    """

    files = glob.glob(pattern)
    li = []

    for f in files:
        if filetype == "excel":
            df = pl.read_excel(f)
        elif filetype == "csv":
            df = pl.read_csv(f)
        li.append(df)
        del df
        gc.collect()

    return pl.concat(li, how="diagonal")


def load_keywords(kw_file="/home/msalvetti/notebooks_2/data/raw/keywords.csv"):
    """
    Loads and cleans up keyword file
    """

    keywords = pl.read_csv(kw_file)
    keywords = (
        keywords.with_columns(pl.col("prompt_output").str.split(", "))
        .explode("prompt_output")
        .filter(pl.col("prompt_output").str.len_chars() > 4)
        .unique()
        .group_by("category")
        .agg(pl.col("prompt_output").str.join(", "))
    )
    return keywords


def load_parquets(path: str) -> pl.LazyFrame:
    """
    Function that safely loads all parquet files in the specified folder, excluding corrupted ones, and return a polars LazyFrame
    Args:
    path (str): Folder where the parquet files are
    """
    corrupted = []

    for f in Path(path).glob("*.parquet"):
        try:
            pf = pq.ParquetFile(f)
            schema = pf.schema_arrow
        except Exception:
            print(f"File {f.name} is corrupted")
            corrupted.append(f)

    all_files = [f for f in Path(path).glob("*.parquet")]
    files = [f for f in all_files if f not in corrupted]

    return pl.scan_parquet(files)


def load_keyword_matches(filename: str) -> pl.DataFrame:
    from config import DATA_DIR

    df = pl.read_parquet(DATA_DIR + "/processed/" + filename)
    df = (
        df.filter(pl.col("matches") != [])
        .explode("matches")
        .rename({"matches": "keyword"})
    )
    return df
