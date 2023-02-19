## Speed Comparison

TO BE UPDATED

We performed a speed test, comparing [retriv](https://github.com/AmenRa/retriv) to [rank_bm25](https://github.com/dorianbrown/rank_bm25), a popular [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) implementation in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), and [pyserini](https://github.com/castorini/pyserini), a [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) binding to the [Lucene](https://en.wikipedia.org/wiki/Apache_Lucene) search engine.

We relied on the [MSMARCO Passage](https://microsoft.github.io/msmarco) dataset to collect documents and queries.
Specifically, we used the original document collection and three sub-samples of it, accounting for 1k, 100k, and 1M documents, respectively, and sampled 1k queries from the original ones.
We computed the top-100 results with each library (if possible). 
Results are reported below. Best results are highlighted in boldface.

| Library                                                   | Collection Size | Elapsed Time | Avg. Query Time | Throughput (q/s) |
| --------------------------------------------------------- | --------------: | -----------: | --------------: | ---------------: |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25)     |           1,000 |        646ms |           6.5ms |           1548/s |
| [pyserini](https://github.com/castorini/pyserini)         |           1,000 |      1,438ms |           1.4ms |            695/s |
| [retriv](https://github.com/AmenRa/retriv)                |           1,000 |        140ms |           0.1ms |           7143/s |
| [retriv](https://github.com/AmenRa/retriv) (multi-search) |           1,000 |    __134ms__ |       __0.1ms__ |       __7463/s__ |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25)     |         100,000 |    106,000ms |          1060ms |              1/s |
| [pyserini](https://github.com/castorini/pyserini)         |         100,000 |      2,532ms |           2.5ms |            395/s |
| [retriv](https://github.com/AmenRa/retriv)                |         100,000 |        314ms |           0.3ms |           3185/s |
| [retriv](https://github.com/AmenRa/retriv) (multi-search) |         100,000 |    __256ms__ |       __0.3ms__ |       __3906__/s |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25)     |       1,000,000 |          N/A |             N/A |              N/A |
| [pyserini](https://github.com/castorini/pyserini)         |       1,000,000 |      4,060ms |           4.1ms |            246/s |
| [retriv](https://github.com/AmenRa/retriv)                |       1,000,000 |      1,018ms |           1.0ms |            982/s |
| [retriv](https://github.com/AmenRa/retriv) (multi-search) |       1,000,000 |    __503ms__ |       __0.5ms__ |       __1988/s__ |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25)     |       8,841,823 |          N/A |             N/A |              N/A |
| [pyserini](https://github.com/castorini/pyserini)         |       8,841,823 |     12,245ms |          12.2ms |             82/s |
| [retriv](https://github.com/AmenRa/retriv)                |       8,841,823 |     10,763ms |          10.8ms |             93/s |
| [retriv](https://github.com/AmenRa/retriv) (multi-search) |       8,841,823 |  __4,476ms__ |       __4.4ms__ |        __227/s__ |