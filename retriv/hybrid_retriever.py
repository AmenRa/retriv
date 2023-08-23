from typing import Dict, Iterable, List, Set, Union

import numpy as np
import orjson
from oneliner_utils import create_path
from tqdm import tqdm

from .base_retriever import BaseRetriever
from .dense_retriever.dense_retriever import DenseRetriever
from .merger.merger import Merger
from .paths import hr_state_path
from .sparse_retriever.sparse_retriever import SparseRetriever


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        # Global params
        index_name: str = "new-index",
        # Sparse retriever params
        sr_model: str = "bm25",
        min_df: int = 1,
        tokenizer: Union[str, callable] = "whitespace",
        stemmer: Union[str, callable] = "english",
        stopwords: Union[str, List[str], Set[str]] = "english",
        do_lowercasing: bool = True,
        do_ampersand_normalization: bool = True,
        do_special_chars_normalization: bool = True,
        do_acronyms_normalization: bool = True,
        do_punctuation_removal: bool = True,
        # Dense retriever params
        dr_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        max_length: int = 128,
        use_ann: bool = True,
        # For already instantiated modules
        sparse_retriever: SparseRetriever = None,
        dense_retriever: DenseRetriever = None,
        merger: Merger = None,
    ):
        """The [Hybrid Retriever](https://github.com/AmenRa/retriv/blob/main/docs/hybrid_retriever.md) is searcher based on both lexical and semantic matching. It comprises three components: the [Sparse Retriever]((https://github.com/AmenRa/retriv/blob/main/docs/sparse_retriever.md)), the [Dense Retriever]((https://github.com/AmenRa/retriv/blob/main/docs/dense_retriever.md)), and the Merger. The Merger fuses the results of the Sparse and Dense Retrievers to compute the _hybrid_ results.

        Args:
            index_name (str, optional): [retriv](https://github.com/AmenRa/retriv) will use `index_name` as the identifier of your index. Defaults to "new-index".

            sr_model (str, optional): defines the model to use for sparse retrieval (`bm25` or `tf-idf`). Defaults to "bm25".

            min_df (int, optional): terms that appear in less than `min_df` documents will be ignored. If integer, the parameter indicates the absolute count. If float, it represents a proportion of documents. Defaults to 1.

            tokenizer (Union[str, callable], optional): [tokenizer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable tokenizer or disable tokenization setting the parameter to `None`. Defaults to "whitespace".

            stemmer (Union[str, callable], optional): [stemmer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable stemmer or disable stemming setting the parameter to `None`. Defaults to "english".

            stopwords (Union[str, List[str], Set[str]], optional): [stopwords](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to remove during preprocessing. You can pass a custom stop-word list or disable stop-words removal by setting the parameter to `None`. Defaults to "english".

            do_lowercasing (bool, optional): whether to lowercase texts. Defaults to True.

            do_ampersand_normalization (bool, optional): whether to convert `&` in `and` during pre-processing. Defaults to True.

            do_special_chars_normalization (bool, optional): whether to remove special characters for letters, e.g., `übermensch` → `ubermensch`. Defaults to True.

            do_acronyms_normalization (bool, optional): whether to remove full stop symbols from acronyms without splitting them in multiple words, e.g., `P.C.I.` → `PCI`. Defaults to True.

            do_punctuation_removal (bool, optional): whether to remove punctuation. Defaults to True.

            dr_model (str, optional): defines the model to use for encoding queries and documents into vectors. You can use an [HuggingFace's Transformers](https://huggingface.co/models) pre-trained model by providing its ID or load a local model by providing its path. In the case of local models, the path must point to the directory containing the data saved with the [`PreTrainedModel.save_pretrained`](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) method. Note that the representations are computed with `mean pooling` over the `last_hidden_state`. Defaults to "sentence-transformers/all-MiniLM-L6-v2".

            normalize (bool, optional):  whether to L2 normalize the vector representations. Defaults to True.

            max_length (int, optional): texts longer than `max_length` will be automatically truncated. Choose this parameter based on how the employed model was trained or is generally used. Defaults to 128.

            use_ann (bool, optional): whether to use approximate nearest neighbors search. Set it to `False` to use nearest neighbors search without approximation. If you have less than 20k documents in your collection, you probably want to disable approximation. Defaults to True.
        """

        self.index_name = index_name

        self.sparse_retriever = (
            sparse_retriever
            if sparse_retriever is not None
            else SparseRetriever(
                index_name=index_name,
                model=sr_model,
                min_df=min_df,
                tokenizer=tokenizer,
                stemmer=stemmer,
                stopwords=stopwords,
                do_lowercasing=do_lowercasing,
                do_ampersand_normalization=do_ampersand_normalization,
                do_special_chars_normalization=do_special_chars_normalization,
                do_acronyms_normalization=do_acronyms_normalization,
                do_punctuation_removal=do_punctuation_removal,
            )
        )

        self.dense_retriever = (
            dense_retriever
            if dense_retriever is not None
            else DenseRetriever(
                index_name=index_name,
                model=dr_model,
                normalize=normalize,
                max_length=max_length,
                use_ann=use_ann,
            )
        )

        self.merger = merger if merger is not None else Merger(index_name=index_name)

    def index(
        self,
        collection: Iterable,
        embeddings_path: str = None,
        use_gpu: bool = False,
        batch_size: int = 512,
        callback: callable = None,
        show_progress: bool = True,
    ):
        """Index a given collection of documents.

        Args:
            collection (Iterable): collection of documents to index.

            embeddings_path (str, optional): in case you want to load pre-computed embeddings, you can provide the path to a `.npy` file. Embeddings must be in the same order as the documents in the collection file. Defaults to None.

            use_gpu (bool, optional): whether to use the GPU for document encoding. Defaults to False.

            batch_size (int, optional): how many documents to encode at once. Regulate it if you ran into memory usage issues or want to maximize throughput. Defaults to 512.

            callback (callable, optional): callback to apply before indexing the documents to modify them on the fly if needed. Defaults to None.

            show_progress (bool, optional): whether to show a progress bar for the indexing process. Defaults to True.

        Returns:
            HybridRetriever: Hybrid Retriever
        """

        self.save_collection(collection, callback)

        self.initialize_doc_index()
        self.initialize_id_mapping()
        self.doc_count = len(self.id_mapping)

        # Sparse ---------------------------------------------------------------
        self.sparse_retriever.doc_index = self.doc_index
        self.sparse_retriever.id_mapping = self.id_mapping
        self.sparse_retriever.doc_count = self.doc_count
        self.sparse_retriever.index_aux(show_progress)

        # Dense ----------------------------------------------------------------
        self.dense_retriever.doc_index = self.doc_index
        self.dense_retriever.id_mapping = self.id_mapping
        self.dense_retriever.doc_count = self.doc_count
        self.dense_retriever.index_aux(
            embeddings_path, use_gpu, batch_size, callback, show_progress
        )

        self.save()

        return self

    def index_file(
        self,
        path: str,
        embeddings_path: str = None,
        use_gpu: bool = False,
        batch_size: int = 512,
        callback: callable = None,
        show_progress: bool = True,
    ):
        """Index the collection contained in a given file.

        Args:
            path (str): path of file containing the collection to index.

            embeddings_path (str, optional): in case you want to load pre-computed embeddings, you can provide the path to a `.npy` file. Embeddings must be in the same order as the documents in the collection file. Defaults to None.

            use_gpu (bool, optional): whether to use the GPU for document encoding. Defaults to False.

            batch_size (int, optional): how many documents to encode at once. Regulate it if you ran into memory usage issues or want to maximize throughput. Defaults to 512.

            callback (callable, optional): callback to apply before indexing the documents to modify them on the fly if needed. Defaults to None.

            show_progress (bool, optional): whether to show a progress bar for the indexing process. Defaults to True.

        Returns:
            HybridRetriever: Hybrid Retriever.
        """

        collection = self.collection_generator(path, callback)
        return self.index(
            collection,
            embeddings_path,
            use_gpu,
            batch_size,
            None,
            show_progress,
        )

    def save(self):
        """Save the state of the retriever to be able to restore it later."""

        state = dict(
            id_mapping=self.id_mapping,
            doc_count=self.doc_count,
        )
        np.savez_compressed(hr_state_path(self.index_name), state=state)

        self.sparse_retriever.save()
        self.dense_retriever.save()
        self.merger.save()

    @staticmethod
    def load(index_name: str = "new-index"):
        """Load a retriever and its index.

        Args:
            index_name (str, optional): Name of the index. Defaults to "new-index".

        Returns:
            HybridRetriever: Hybrid Retriever.
        """

        state = np.load(hr_state_path(index_name), allow_pickle=True)["state"][()]

        hr = HybridRetriever(index_name)
        hr.initialize_doc_index()
        hr.id_mapping = state["id_mapping"]
        hr.doc_count = state["doc_count"]

        hr.sparse_retriever = SparseRetriever.load(index_name)
        hr.dense_retriever = DenseRetriever.load(index_name)
        hr.merger = Merger.load(index_name)
        return hr

    def search(
        self,
        query: str,
        return_docs: bool = True,
        cutoff: int = 100,
    ) -> List:
        """Standard search functionality.

        Args:
            query (str): what to search for.

            return_docs (bool, optional): whether to return the texts of the documents. Defaults to True.

            cutoff (int, optional): number of results to return. Defaults to 100.

        Returns:
            List: results.
        """

        sparse_results = self.sparse_retriever.search(query, False, 1_000)
        dense_results = self.dense_retriever.search(query, False, 1_000)
        hybrid_results = self.merger.fuse([sparse_results, dense_results])
        return (
            self.prepare_results(
                list(hybrid_results.keys())[:cutoff],
                list(hybrid_results.values())[:cutoff],
            )
            if return_docs
            else hybrid_results
        )

    def msearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
        batch_size: int = 32,
    ) -> Dict:
        """Compute results for multiple queries at once.

        Args:
            queries (List[Dict[str, str]]): what to search for.

            cutoff (int, optional): number of results to return. Defaults to 100.

            batch_size (int, optional): how many queries to search at once. Regulate it if you ran into memory usage issues or want to maximize throughput. Defaults to 32.

        Returns:
            Dict: results.
        """

        sparse_results = self.sparse_retriever.msearch(queries, 1_000)
        dense_results = self.dense_retriever.msearch(queries, 1_000, batch_size)
        return self.merger.mfuse([sparse_results, dense_results], cutoff)

    def bsearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
        batch_size: int = 32,
        show_progress: bool = True,
        qrels: Dict[str, Dict[str, float]] = None,
        path: str = None,
    ):
        """Batch-Search is similar to Multi-Search but automatically generates batches of queries to evaluate and allows dynamic writing of the search results to disk in [JSONl](https://jsonlines.org) format. bsearch is handy for computing results for hundreds of thousands or even millions of queries without hogging your RAM.

        Args:
            queries (List[Dict[str, str]]): what to search for.

            cutoff (int, optional): number of results to return. Defaults to 100.

            batch_size (int, optional): how many queries to search at once. Regulate it if you ran into memory usage issues or want to maximize throughput. Defaults to 32.

            show_progress (bool, optional): whether to show a progress bar for the search process. Defaults to True.

            qrels (Dict[str, Dict[str, float]], optional): query relevance judgements for the queries. Defaults to None.

            path (str, optional): where to save the results. Defaults to None.

        Returns:
            Dict: results.
        """

        batches = [
            queries[i : i + batch_size] for i in range(0, len(queries), batch_size)
        ]

        results = {}

        pbar = tqdm(
            total=len(queries),
            disable=not show_progress,
            desc="Batch search",
            dynamic_ncols=True,
            mininterval=0.5,
        )

        if path is None:
            for batch in batches:
                new_results = self.msearch(
                    queries=batch, cutoff=cutoff, batch_size=len(batch)
                )
                results = {**results, **new_results}
                pbar.update(min(batch_size, len(batch)))
        else:
            path = create_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "wb") as f:
                for batch in batches:
                    new_results = self.msearch(queries=batch, cutoff=cutoff)

                    for i, (k, v) in enumerate(new_results.items()):
                        x = {
                            "id": k,
                            "text": batch[i]["text"],
                            "hybrid_doc_ids": list(v.keys()),
                            "hybrid_scores": [float(s) for s in list(v.values())],
                        }
                        if qrels is not None:
                            x["rel_doc_ids"] = list(qrels[k].keys())
                            x["rel_scores"] = list(qrels[k].values())
                        f.write(orjson.dumps(x) + "\n".encode())

                    pbar.update(min(batch_size, len(batch)))

        return results

    def autotune(
        self,
        queries: List[Dict[str, str]],
        qrels: Dict[str, Dict[str, float]],
        metric: str = "ndcg",
        n_trials: int = 100,
        cutoff: int = 100,
        batch_size: int = 32,
    ):
        """Use the AutoTune function to tune the Sparse Retriever's model [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) parameters and the importance given to the lexical and semantic relevance scores computed by the Sparse and Dense Retrievers, respectively. All metrics supported by [ranx](https://github.com/AmenRa/ranx) are supported by the `autotune` function. At the of the process, the best parameter configurations are automatically applied and saved to disk. You can inspect the best configurations found by printing `hr.sparse_retriever.hyperparams`, `hr.merger.norm` and `hr.merger.params`.

        Args:
            queries (List[Dict[str, str]]): queries to use for the optimization process.

            qrels (Dict[str, Dict[str, float]]): query relevance judgements for the queries.

            metric (str, optional): metric to optimize for. Defaults to "ndcg".

            n_trials (int, optional): number of configuration to evaluate. Defaults to 100.

            cutoff (int, optional): number of results to consider for the optimization process. Defaults to 100.
        """

        # Tune sparse ----------------------------------------------------------
        self.sparse_retriever.autotune(
            queries=queries,
            qrels=qrels,
            metric=metric,
            n_trials=n_trials,
            cutoff=cutoff,
        )

        # Tune merger ----------------------------------------------------------
        sparse_results = self.sparse_retriever.msearch(queries, 1_000)
        dense_results = self.dense_retriever.msearch(queries, 1_000, batch_size)
        self.merger.autotune(qrels, [sparse_results, dense_results], metric)
