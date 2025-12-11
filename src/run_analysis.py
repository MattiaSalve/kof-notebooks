import polars as pl


def compute_errors_percent(lf: pl.LazyFrame) -> pl.DataFrame:
    n_companies = (
        lf.filter(pl.col("dl_rank") == 0).select(pl.col("ID").len()).collect().item()
    )
    errors = (
        lf.filter(pl.col("dl_rank") == 0)
        .group_by("error")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") * 100 / n_companies).round(2).alias("percent"))
        .sort("count", descending=True)
        .collect()
    )
    return errors


def get_subpage_counts(lf: pl.LazyFrame) -> pl.DataFrame:
    subpage_counts = (
        lf.filter(pl.col("error").is_not_null())
        .group_by("ID")
        .agg((pl.len() - 1).alias("count"))
        .collect()
    )
    return subpage_counts.filter(pl.col("count") > 0)
