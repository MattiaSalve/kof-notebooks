from pathlib import Path
import polars as pl

DATA_DIR = "/home/msalvetti/notebooks_2/data/"
KEYWORDS_FILE = "/home/msalvetti/notebooks_2/data/raw/keywords.csv"
PATENTS_DIR = DATA_DIR + "raw/Orbis Swiss Inventor Patent Data/"
CHUNK_DIR = "/home/msalvetti/KOFScraper/chunks/"


def print_longer(df: pl.DataFrame, length: int):
    with pl.Config(set_fmt_str_lengths=length, set_tbl_width_chars=length):
        print(df)
