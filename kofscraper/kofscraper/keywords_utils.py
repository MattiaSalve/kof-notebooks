from pathlib import Path
from typing import List
from datetime import timedelta
import gc, os, time, glob

import ahocorasick
from numpy import vectorize
from pandas import DataFrame
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
import enchant


def find_keywords_ac(text: str, automaton):
    if text is None:
        return []

    text_lower = text.lower()
    matches = []

    for end_pos, original_kw in automaton.iter(text_lower):
        start_pos = end_pos - len(original_kw) + 1

        if start_pos > 0:
            char_before = text_lower[start_pos - 1]
            if char_before.isalnum() or char_before == "_":
                continue

            if end_pos + 1 < len(text_lower):
                char_after = text_lower[end_pos + 1]
                if char_after.isalnum() or char_after == "__":
                    continue

            matches.append(original_kw)
    return matches


def match_keywords(
    keywords: List["str"],
    lf: pl.LazyFrame,
    text_col: str,
    idx_col: str,
    batch_size: int,
    output_name: str,
):
    """
    Gets keyword matches from text chunking the dataframe so that you may read data that is larger than memory
    Args:
        - keywords (List["str"]): the list of words to be matched
        - lf (pl.LazyFrame): DataFrame containing the text
        - text_col (str): Name of the column containing the text
        - idx_col (str): Name of the column containing the unique identifier for the total_rows
        - batch_size (int): Number of rows to be processed at a time
        - output_name (str): Name of the output file

    Returns:
        Nothing, it writes the result to a file.
    """
    print(f"Building automaton with {len(keywords)} keywords")
    automaton = ahocorasick.Automaton()
    for keyword in keywords:
        automaton.add_word(keyword, keyword)
    automaton.make_automaton()
    print("Automaton built")

    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_rows = lf.select(pl.len()).collect(engine="streaming").item()

    output_files = []
    start = time.time()

    for i in range(0, total_rows, batch_size):
        frame = lf.slice(i, batch_size).collect(engine="streaming")

        texts = frame[text_col].to_list()
        idx = frame[idx_col].to_list()

        matches = [find_keywords_ac(t, automaton) for t in texts]
        batch_df = pl.DataFrame(
            {"ID": idx, "matches": matches},
            schema={"ID": pl.String, "matches": pl.List(pl.String)},
        )
        output = OUTPUT_DIR / f"batch_{i:06d}.parquet"
        batch_df.write_parquet(output)
        output_files.append(str(output))
        percent_done = (i + batch_size) / total_rows
        end = time.time()
        eta = ((end - start) / percent_done) - (end - start)
        print(
            f"Percent completed: {percent_done*100:.2f}%, Time remaining: {timedelta(seconds = int(eta))}"
        )
        del texts, idx, matches
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


def rank_keywords(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ranks the keywords by order of appearence, returning
    """
    df = (
        df.group_by("keyword")
        .agg(pl.len())
        .group_by(pl.col("len"))
        .agg(pl.col("keyword").str.join(", "))
        .sort(pl.col("len"), descending=True)
        .with_columns(
            pl.col("len").rank(method="ordinal", descending=True),
            pl.col("keyword").str.split(", "),
        )
    ).explode("keyword")
    return df


def add_jaccard(df: pl.DataFrame, col1: str, col2: str) -> pl.DataFrame:
    return (
        df.with_columns(
            intersection=(pl.col(col1).list.set_intersection(pl.col(col2))).list.len(),
            union=(pl.col(col1).list.set_union(pl.col(col2))).list.len(),
        )
        .with_columns(jaccard_similarity=pl.col("intersection") / pl.col("union"))
        .select(pl.exclude("intersection", "union"))
    )


def find_conditional_probability(
    df: pl.DataFrame, conditional_column: str, other_column: str, category: str
):
    """
    Calculates the conditional probability of a company being active in a category on the website/patent, given the conditional_column condition
    """
    n_intersection = (
        df.select(conditional_column, other_column)
        .with_columns(
            pl.col(conditional_column).list.contains(category),
            pl.col(other_column).list.contains(category),
        )
        .with_columns(match=pl.col(conditional_column) & pl.col(other_column))
        .filter(pl.col("match") == 1)
        .select(pl.len())
        .item()
    )

    n_conditional = (
        df.select(conditional_column)
        .with_columns(pl.col(conditional_column).list.contains(category))
        .filter(pl.col(conditional_column) == True)
        .select(pl.len())
        .item()
    )

    if n_conditional > 0:
        return n_intersection / n_conditional
    else:
        return 0


def get_companies_with_n_unique_kws(df: pl.DataFrame, thresh: int) -> pl.DataFrame:
    """
    Returns a dataframe with companies and the list of categories that they match with thresh or more unique keywords
    """
    df = (
        (
            df.group_by("ID", "category")
            .agg(pl.col("keyword").n_unique().alias("n_unique"))
            .filter(pl.col("n_unique") >= thresh)
        )
        .group_by("ID")
        .agg(pl.col("category"))
    ).with_columns(n_unique=pl.col("category").list.len())
    return df


def get_keyword_decay(df: pl.DataFrame, category: str) -> pl.DataFrame:
    return (
        df.filter(pl.col("category") == category)
        .group_by("n_unique")
        .agg(pl.len())
        .sort("n_unique")
        .with_columns(pl.col("len").cum_sum(reverse=True))
    )


def get_tf_idf_language(
    lf: pl.LazyFrame, text_col: str, chunk_size: int, output_name: str
):
    """
    Gets the tf-idf score for all words on the given column, using chunking to process larger than memory datasets.
    Please note that the tf-idf score will then be chunk based, so you need to use a big enough chunk size to reduce outliars.

    Returns:
    A dataframe with word | tf_idf score | columns with true or false depending on wether a word is part of that language (en, de, fr, and it only)
    """

    lf = lf.filter(pl.col(text_col).is_not_null())
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    output_files = []

    start = time.time()

    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    for i in range(0, n_rows, chunk_size):
        vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", min_df=2, max_df=0.95
        )
        corpus = (
            lf.slice(i, chunk_size)
            .select(text_col)
            .collect(engine="streaming")[text_col]
            .to_list()
        )
        X = vectorizer.fit_transform(corpus)
        features = vectorizer.get_feature_names_out()

        english = enchant.Dict("en_US")
        german = enchant.Dict("de_DE")
        french = enchant.Dict("fr_FR")
        italian = enchant.Dict("it-IT")

        en = []
        de = []
        fr = []
        it = []
        for w in features:
            en.append(english.check(w))
            de.append(german.check(w))
            fr.append(french.check(w))
            it.append(italian.check(w))

        chunk_tf_idf = pl.DataFrame(
            {
                "word": features,
                "tf_idf": vectorizer.idf_,
                "en": en,
                "de": de,
                "fr": fr,
                "it": it,
            }
        )

        output = OUTPUT_DIR / f"batch_{i:09d}.parquet"
        chunk_tf_idf.write_parquet(output)
        output_files.append(str(output))

        percent_done = (i + chunk_size) / n_rows
        end = time.time()
        eta = ((end - start) / percent_done) - (end - start)
        print(
            f"Percent completed: {percent_done*100:.2f}%, Time remaining: {timedelta(seconds = int(eta))}"
        )
        del X, features, chunk_tf_idf
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
