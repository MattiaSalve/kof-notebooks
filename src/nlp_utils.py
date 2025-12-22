from pathlib import Path
import gc, os, torch, glob
import polars as pl
import numpy as np

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util


def perform_chunked_function(lf: pl.LazyFrame, chunk_size: int, function, *args):
    """
    Perform the specified function on a LazyFrame by chunks, so that it can be executed on larger than memory data.
    Saves the resulting dataframe to "result.parquet"
    Args:
        lf (pl.LazyFrame): Lazy frame to operate the function on.
        chunk_size (int): Number of rows to operate the function on at once.
        function (function): Function to be performed.
        *args: Any number of argument that the function will take as inputs.

    Return:
        Nothing, saves the result to a "result.parquet".
    """

    n_rows = lf.select(pl.len()).collect(engine="streaming").item()

    DIR = Path("temp")
    DIR.mkdir(exist_ok=True)

    for i in range(0, n_rows, chunk_size):
        frame = lf.slice(i, chunk_size).collect(engine="streaming")

        result = function(frame, args)
        result.write_parquet(DIR / f"result_{i}.parquet")

        del frame, result
        gc.collect()

    final = pl.scan_parquet(DIR / "*.parquet")
    final.sink_parquet("result.parquet")

    for f in glob.glob(str(DIR) + "/*"):
        os.remove(f)
    DIR.rmdir()


def get_embeddings(
    sentences, model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> np.ndarray:
    """
    Calculates the embeddings for a (list of) text(s) and return the embeddings using the specified model
    Args:
        sentences (List(str)): Texts to convert to embeddings.
        model_id (str): Model to use.
    Returns:
        np.ndarray: array of arrays containing the embeddings for each sentence passed
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = model_output.last_hidden_state[:, 0, :]
    return sentence_embeddings.numpy()


def get_cos_similarity(emb1, emb2) -> np.ndarray:
    """
    Calculates the cosine similarity between two arrays of embeddings
    Args:
        emb1: First list of embeddings.
        emb2: Second list of embeddings.
    Returns:
        np.ndarray: array of similarities with each element of length len(emb2)
    """
    return util.pytorch_cos_sim(emb1, emb2).numpy()
