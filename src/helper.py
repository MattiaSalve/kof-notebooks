import pandas as pd
import polars as pl
import json
import ast
from io import StringIO


def read_json_files(path):
    data = pl.read_ndjson(path)
    data=data.with_columns(pl.col("custom_id").str.slice(0, 14).alias("ID"))
    data = (
        data.select(["ID","response"])
        .unnest("response")
        .unnest("body")
        .select("choices", "ID")
        .with_columns(pl.col("choices").list.get(0))
        .unnest("choices")
        .select(["ID", "message"])
        .unnest("message")
        .select("ID", "content")
        .map_rows(lambda v: (_check_json(v[1]), v[0]))
        .unnest("column_0")
        .rename({"column_1": "ID"})
    )
    return data

def _check_json(s):
    if s is None:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None