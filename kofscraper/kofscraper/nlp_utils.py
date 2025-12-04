from pathlib import Path
from typing import List
from datetime import timedelta
import gc, os, time, glob

import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from pandas import DataFrame
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def get_embeddings(lf: pl.LazyFrame, text_col: str, chunk_size: int, output_name: str):
    tokenizer = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()

    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    output_files = []
    start = time.time()

    for i in range(0, n_rows, chunk_size):
        frame = lf.slice(i, chunk_size).collect(engine="streaming")
        texts = frame.select(pl.col(text_col))[text_col].to_list()
        urls = frame.select(pl.col("url"))
        tokens = tokenizer.encode(texts)

        output = OUTPUT_DIR / f"chunk_{i:09d}.parquet"
        chunk_emb = pl.DataFrame({"url": urls, "embeddings": tokens})
        chunk_emb.write_parquet(output)
        output_files.append(str(output))

        percent_done = (i + chunk_size) / n_rows
        end = time.time()
        eta = ((end - start) / percent_done) - (end - start)
        print(
            f"Percent completed: {percent_done*100:.2f}%, Time remaining: {timedelta(seconds = int(eta))}"
        )
        del texts, chunk_emb, frame
        gc.collect()
    final_result = pl.scan_parquet(output_files)
    name = output_name + ".parquet"
    final_result.sink_parquet(Path("/home/msalvetti/notebooks_2/data/processed") / name)
    print(f"Wrote results to 'data/processed/{output_name}.parquet'")

    print("Cleaning up files...")
    for f in output_files:
        os.remove(f)
    OUTPUT_DIR.rmdir()
    print("Done.")


def get_similarity(emb_frame, queries, model):

    for query in queries:
        emb_query = model.encode(query)
        sim = cos_sim(emb_query, emb_frame["embeddings"]).cpu().tolist()[0]
        emb_frame = emb_frame.with_columns(
            pl.Series(f"sim_{query}", sim, dtype=pl.Float32)
        )
    return emb_frame


def get_similarity_to_kws(emb_frame, kws_df: pl.DataFrame, model):

    for query in kws_df.iter_rows():
        emb_query = model.encode(query[1])
        sim = cos_sim(emb_query, emb_frame["embeddings"]).cpu().tolist()[0]
        emb_frame = emb_frame.with_columns(
            pl.Series(f"sim_{query[0]}", sim, dtype=pl.Float32)
        )
    return emb_frame


def get_categories(emb_frame: pl.DataFrame, thresh: float):

    sim_cols = [c for c in emb_frame.columns if c.startswith("sim_")]

    emb_frame = (
        emb_frame.with_columns(
            pl.concat_list(
                [
                    pl.when(pl.col(c) > thresh)
                    .then(pl.lit(name.replace("sim_", "")))
                    .otherwise(pl.lit(None))
                    for c in sim_cols
                    for name in [c]
                ]
            )
            .list.eval(pl.element().drop_nulls())
            .alias("tmp")
        )
        .with_columns(
            predicted=pl.when(pl.col("tmp").list.len() == 0)
            .then(pl.lit(["unknown"]))
            .otherwise(pl.col("tmp"))
        )
        .drop("tmp")
    )
    return emb_frame


def tokenize_batch(texts, max_len=512):
    MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
    )

    return {
        "input_ids": np.array(enc["input_ids"], dtype=np.int32),
        "attention_mask": np.array(enc["attention_mask"], dtype=np.int32),
        "token_type_ids": np.array(enc.get("token_type_ids", []), dtype=np.int32),
    }


def write_parquet_tokenized(path, ids, masks, token_types=None, meta=None):
    arrays = [
        pa.array(ids.tolist(), type=pa.list_(pa.int32())),
        pa.array(masks.tolist(), type=pa.list_(pa.int32())),
    ]
    names = ["input_ids", "attention_mask"]
    if token_types is not None and len(token_types):
        arrays.append(pa.array(token_types.tolist(), type=pa.list_(pa.int32())))
        names.append("token_type_ids")

    table = pa.Table.from_arrays(arrays, names=names)
    if meta:  # store model id and tokenizer config
        table = table.replace_schema_metadata({k: str(v) for k, v in meta.items()})
    pq.write_table(table, path, compression="zstd")


def get_tokens(lf: pl.LazyFrame, chunk_size: int, output_name: str):
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()

    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_files = []
    start = time.time()

    for i in range(0, n_rows, chunk_size):
        chunk = lf.slice(i, chunk_size).collect(engine="streaming")["text"].to_list()
        toks = tokenize_batch(chunk)
        output = OUTPUT_DIR / f"chunk_{i:09d}.parquet"
        write_parquet_tokenized(output, toks["input_ids"], toks["attention_mask"])
        output_files.append(output)

        percent_done = (i + chunk_size) / n_rows
        end = time.time()
        eta = ((end - start) / percent_done) - (end - start)
        print(
            f"Percent completed: {percent_done*100:.2f}%, Time remaining: {timedelta(seconds = int(eta))}"
        )
        del chunk, toks
        gc.collect()

    final_result = pl.scan_parquet(output_files)
    name = output_name + ".parquet"
    final_result.sink_parquet(Path("/home/msalvetti/notebooks_2/data/processed") / name)
    print(f"Wrote results to 'data/processed/{output_name}.parquet'")

    print("Cleaning up files...")
    for f in output_files:
        os.remove(f)
    OUTPUT_DIR.rmdir()
    print("Done.")
