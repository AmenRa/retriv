from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def convert_df_matrix_into_inverted_index(
    df_matrix, vocabulary, show_progress: bool = True
):
    inverted_index = defaultdict(dict)

    for i, term in enumerate(
        tqdm(
            vocabulary,
            disable=not show_progress,
            desc="Building inverted index",
            dynamic_ncols=True,
            mininterval=0.5,
        )
    ):
        inverted_index[term]["doc_ids"] = df_matrix[i].indices
        inverted_index[term]["tfs"] = df_matrix[i].data

    return inverted_index


def build_inverted_index(
    collection: Iterable,
    n_docs: int,
    min_df: int = 1,
    show_progress: bool = True,
) -> Dict:
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        min_df=min_df,
        dtype=np.int16,
    )

    # [n_docs x n_terms]
    df_matrix = vectorizer.fit_transform(
        tqdm(
            collection,
            total=n_docs,
            disable=not show_progress,
            desc="Building TDF matrix",
            dynamic_ncols=True,
            mininterval=0.5,
        )
    )
    # [n_terms x n_docs]
    df_matrix = df_matrix.transpose().tocsr()
    vocabulary = vectorizer.get_feature_names_out()
    inverted_index = convert_df_matrix_into_inverted_index(
        df_matrix=df_matrix,
        vocabulary=vocabulary,
        show_progress=show_progress,
    )

    doc_lens = np.squeeze(np.asarray(df_matrix.sum(axis=0), dtype=np.float32))
    relative_doc_lens = doc_lens / np.mean(doc_lens, dtype=np.float32)

    return dict(inverted_index), relative_doc_lens
