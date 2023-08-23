from collections import defaultdict

import pytest

from retriv.experimental import AdvancedRetriever


# FIXTURES =====================================================================
@pytest.fixture
def schema():
    return {
        "id": "id",
        "lyrics": "text",
        "year": "number",
        "ozzy": "bool",
        "album": "keyword",
        "genre": "keywords",
    }


@pytest.fixture
def collection():
    return [
        {
            "id": "doc_0",
            "lyrics": "Generals gathered in their masses",
            "album": "Black Sabbath",
            "year": 1969,
            "ozzy": True,
            "genre": ["Doom", "Heavy Metal"],
        },
        {
            "id": "doc_1",
            "lyrics": "Just like witches at black masses",
            "album": "Paranoid",
            "year": 1970,
            "ozzy": True,
            "genre": ["Doom", "Heavy Metal"],
        },
        {
            "id": "doc_2",
            "lyrics": "Evil minds that plot destruction",
            "album": "Heaven and Hell",
            "year": 1980,
            "ozzy": False,
            "genre": ["Heavy Metal"],
        },
    ]


def test_check_schema_no_id():
    with pytest.raises(Exception, match="Schema must contain an id field"):
        AdvancedRetriever({"text": "text"})


def test_check_schema_invalid_key():
    with pytest.raises(Exception, match="Schema keys must be strings"):
        AdvancedRetriever({"id": "id", 1: "text"})


def test_check_schema_invalid_value():
    with pytest.raises(Exception, match="Type invalid not supported"):
        AdvancedRetriever({"id": "id", "text": "invalid"})


def test_check_schema_double_text():
    with pytest.raises(Exception, match="Only one field can be text"):
        AdvancedRetriever({"id": "id", "title": "text", "body": "text"})


def test_check_schema_pass(schema):
    se = AdvancedRetriever(schema)
    assert se.schema == schema


def test_check_collection_field_no_id(schema):
    collection = [
        {
            "lyrics": "Generals gathered in their masses",
            "album": "Black Sabbath",
            "year": 1969,
            "ozzy": True,
            "genre": ["Doom", "Heavy Metal"],
        }
    ]
    se = AdvancedRetriever(schema)
    with pytest.raises(Exception, match="has no id"):
        se.check_collection(collection, schema)


def test_check_collection_missing_field(schema):
    collection = [
        {
            "id": "doc_0",
            "lyrics": "Generals gathered in their masses",
            "year": 1969,
            "ozzy": True,
            "genre": ["Doom", "Heavy Metal"],
        }
    ]
    se = AdvancedRetriever(schema)
    with pytest.raises(Exception, match="Field album not in doc"):
        se.check_collection(collection, schema)


def test_check_collection_additional_field(schema):
    collection = [
        {
            "id": "doc_0",
            "lyrics": "Generals gathered in their masses",
            "album": "Black Sabbath",
            "year": 1969,
            "ozzy": True,
            "genre": ["Doom", "Heavy Metal"],
            "invalid": "value",
        }
    ]
    se = AdvancedRetriever(schema)
    with pytest.raises(Exception, match="Field invalid not in schema"):
        se.check_collection(collection, schema)


def test_check_collection_field_id_wrong_type(schema):
    collection = [
        {
            "id": [0],
            "lyrics": "Generals gathered in their masses",
            "album": "Black Sabbath",
            "year": 1969,
            "ozzy": True,
            "genre": ["Doom", "Heavy Metal"],
        },
    ]
    se = AdvancedRetriever(schema)
    with pytest.raises(Exception, match="Field id"):
        se.check_collection(collection, schema)


def test_check_collection_field_lyrics_wrong_type(schema):
    collection = [
        {
            "id": "doc_0",
            "lyrics": 666,
            "album": "Black Sabbath",
            "year": 1969,
            "ozzy": True,
            "genre": ["Doom", "Heavy Metal"],
        },
    ]
    se = AdvancedRetriever(schema)
    with pytest.raises(Exception, match="Field lyrics"):
        se.check_collection(collection, schema)


def test_check_collection_field_year_wrong_type(schema):
    collection = [
        {
            "id": "doc_0",
            "lyrics": "Generals gathered in their masses",
            "album": "Black Sabbath",
            "year": "1969",
            "ozzy": True,
            "genre": ["Doom", "Heavy Metal"],
        },
    ]
    se = AdvancedRetriever(schema)
    with pytest.raises(Exception, match="Field year"):
        se.check_collection(collection, schema)


def test_check_collection_field_ozzy_wrong_type(schema):
    collection = [
        {
            "id": "doc_0",
            "lyrics": "Generals gathered in their masses",
            "album": "Black Sabbath",
            "year": 1969,
            "ozzy": "True",
            "genre": ["Doom", "Heavy Metal"],
        },
    ]
    se = AdvancedRetriever(schema)
    with pytest.raises(Exception, match="Field ozzy"):
        se.check_collection(collection, schema)


def test_check_collection_field_album_wrong_type(schema):
    collection = [
        {
            "id": "doc_0",
            "lyrics": "Generals gathered in their masses",
            "album": ["Black Sabbath"],
            "year": 1969,
            "ozzy": True,
            "genre": ["Doom", "Heavy Metal"],
        },
    ]
    se = AdvancedRetriever(schema)
    with pytest.raises(Exception, match="Field album"):
        se.check_collection(collection, schema)


def test_check_collection_field_genre_wrong_type(schema):
    collection = [
        {
            "id": "doc_0",
            "lyrics": "Generals gathered in their masses",
            "album": "Black Sabbath",
            "year": 1969,
            "ozzy": True,
            "genre": "Doom",
        },
    ]
    se = AdvancedRetriever(schema)
    with pytest.raises(Exception, match="Field genre"):
        se.check_collection(collection, schema)


def test_check_collection(collection, schema):
    se = AdvancedRetriever(schema)
    assert se.check_collection(collection, schema) == True


def initialize_metadata(schema):
    se = AdvancedRetriever(schema)
    metadata = se.initialize_metadata(schema)

    assert metadata == {
        "year": [],
        "ozzy": {True: [], False: []},
        "album": defaultdict(list),
        "genre": defaultdict(list),
    }


def test_fill_metadata(schema, collection):
    se = AdvancedRetriever(schema)
    se.metadata = se.initialize_metadata(schema)

    metadata = se.fill_metadata(
        metadata=se.metadata, collection=collection, schema=schema
    )

    assert metadata == {
        "year": [1969, 1970, 1980],
        "ozzy": {True: [0, 1], False: [2]},
        "album": {
            "Black Sabbath": [0],
            "Paranoid": [1],
            "Heaven and Hell": [2],
        },
        "genre": {
            "Doom": [0, 1],
            "Heavy Metal": [0, 1, 2],
        },
    }


def test_index_metadata(collection, schema):
    se = AdvancedRetriever(schema)
    assert se.check_collection(collection, schema) == True

    metadata = se.index_metadata(collection=collection, schema=schema)

    assert len(metadata) == 4
    assert "year" in list(metadata)
    assert "ozzy" in list(metadata)
    assert "album" in list(metadata)
    assert "genre" in list(metadata)

    assert metadata["year"].tolist() == [1969, 1970, 1980]

    assert len(metadata["ozzy"]) == 2
    assert metadata["ozzy"][True].tolist() == [0, 1]
    assert metadata["ozzy"][False].tolist() == [2]

    assert len(metadata["album"]) == 3
    assert metadata["album"]["Black Sabbath"].tolist() == [0]
    assert metadata["album"]["Paranoid"].tolist() == [1]
    assert metadata["album"]["Heaven and Hell"].tolist() == [2]

    assert len(metadata["genre"]) == 2
    assert metadata["genre"]["Doom"].tolist() == [0, 1]
    assert metadata["genre"]["Heavy Metal"].tolist() == [0, 1, 2]


def test_index(collection, schema):
    se = AdvancedRetriever(schema).index(collection)

    assert len(se.doc_ids) == 3


def test_filter_doc_ids_bool_must(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value = "ozzy", "must", True
    assert se.filter_doc_ids(field, clause, value).tolist() == [0, 1]
    field, clause, value = "ozzy", "must", False
    assert se.filter_doc_ids(field, clause, value).tolist() == [2]


def test_filter_doc_ids_bool_must_not(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value = "ozzy", "must not", True
    assert se.filter_doc_ids(field, clause, value).tolist() == [2]
    field, clause, value = "ozzy", "must not", False
    assert se.filter_doc_ids(field, clause, value).tolist() == [0, 1]


def test_filter_doc_ids_keyword_must(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value = "album", "must", "Black Sabbath"
    assert se.filter_doc_ids(field, clause, value).tolist() == [0]
    field, clause, value = "album", "must", "Paranoid"
    assert se.filter_doc_ids(field, clause, value).tolist() == [1]
    field, clause, value = "album", "must", "Heaven and Hell"
    assert se.filter_doc_ids(field, clause, value).tolist() == [2]


def test_filter_doc_ids_keyword_must_multi(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value = "album", "must", ["Black Sabbath", "Heaven and Hell"]
    assert se.filter_doc_ids(field, clause, value).tolist() == [0, 2]


def test_filter_doc_ids_keyword_must_not(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value = "album", "must not", "Black Sabbath"
    assert se.filter_doc_ids(field, clause, value).tolist() == [1, 2]
    field, clause, value = "album", "must not", "Heaven and Hell"
    assert se.filter_doc_ids(field, clause, value).tolist() == [0, 1]


def test_filter_doc_ids_keyword_must_not_multi(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value = "album", "must not", ["Black Sabbath", "Heaven and Hell"]
    assert se.filter_doc_ids(field, clause, value).tolist() == [1]
    field, clause, value = "album", "must not", ["Black Sabbath", "Paranoid"]
    assert se.filter_doc_ids(field, clause, value).tolist() == [2]


def test_filter_doc_ids_number_must(collection, schema):
    se = AdvancedRetriever(schema).index(collection)

    field, clause, value, operator = "year", "must", 1969, "eq"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0]

    field, clause, value, operator = "year", "must", 1969, "gt"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [1, 2]

    field, clause, value, operator = "year", "must", 1969, "gte"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0, 1, 2]

    field, clause, value, operator = "year", "must", 1970, "lt"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0]

    field, clause, value, operator = "year", "must", 1970, "lte"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0, 1]

    field, clause, value, operator = "year", "must", [1970, 1980], "between"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [1, 2]


def test_filter_doc_ids_number_must_not(collection, schema):
    se = AdvancedRetriever(schema).index(collection)

    field, clause, value, operator = "year", "must not", 1969, "eq"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [1, 2]

    field, clause, value, operator = "year", "must not", 1969, "gt"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0]

    field, clause, value, operator = "year", "must not", 1969, "gte"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == []

    field, clause, value, operator = "year", "must not", 1970, "lt"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [1, 2]

    field, clause, value, operator = "year", "must not", 1970, "lte"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [2]

    field, clause, value, operator = "year", "must not", [1970, 1975], "between"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0, 2]


def test_filter_doc_ids_keywords_must_or(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value, operator = "genre", "must", "Doom", "or"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0, 1]


def test_filter_doc_ids_keywords_must_multi_or(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value, operator = "genre", "must", ["Doom", "Heavy Metal"], "or"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0, 1, 2]


def test_filter_doc_ids_keywords_must_not_or(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value, operator = "genre", "must not", "Doom", "or"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [2]


def test_filter_doc_ids_keywords_must_not_multi_or(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value, operator = "genre", "must not", ["Doom"], "or"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [2]
    field, clause, value, operator = "genre", "must not", ["Doom", "Heavy Metal"], "or"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == []


def test_filter_doc_ids_keywords_must_and(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value, operator = "genre", "must", "Doom", "and"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0, 1]


def test_filter_doc_ids_keywords_must_multi_and(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value, operator = "genre", "must", ["Doom", "Heavy Metal"], "and"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [0, 1]


def test_filter_doc_ids_keywords_must_not_and(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value, operator = "genre", "must not", "Doom", "and"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [2]


def test_filter_doc_ids_keywords_must_not_multi_and(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    field, clause, value, operator = "genre", "must not", ["Doom"], "and"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [2]
    field, clause, value, operator = "genre", "must not", ["Doom", "Heavy Metal"], "and"
    assert se.filter_doc_ids(field, clause, value, operator).tolist() == [2]


def test_get_filtered_doc_ids(collection, schema):
    se = AdvancedRetriever(schema).index(collection)

    filters = [
        dict(field="genre", clause="must", value=["Doom", "Heavy Metal"], operator="or")
    ]
    assert se.get_filtered_doc_ids(filters).tolist() == [0, 1, 2]

    filters = [dict(field="year", clause="must not", value=1970, operator="lt")]
    assert se.get_filtered_doc_ids(filters).tolist() == [1, 2]

    filters = [dict(field="ozzy", clause="must", value=True)]
    assert se.get_filtered_doc_ids(filters).tolist() == [0, 1]

    filters = [
        dict(
            field="genre", clause="must", value=["Doom", "Heavy Metal"], operator="or"
        ),  # 0, 1, 2
        dict(field="year", clause="must not", value=1970, operator="lt"),  # 1, 2
        dict(field="ozzy", clause="must", value=True),  # 0, 1
    ]
    assert se.get_filtered_doc_ids(filters).tolist() == [1]


def test_format_filters(collection, schema):
    se = AdvancedRetriever(schema).index(collection)

    filters = {
        "year": ("gte", 1980),
        "ozzy": True,
        "album": ["Paranoid", "Master of Reality"],
        "genre": ("or", ["Doom", "Heavy Metal"]),
    }

    formatted_filters = se.format_filters(filters)
    assert len(formatted_filters) == 4
    assert formatted_filters[0] == dict(
        field="year", clause="must", value=1980, operator="gte"
    )
    assert formatted_filters[1] == dict(field="ozzy", clause="must", value=True)
    assert formatted_filters[2] == dict(
        field="album", clause="must", value=["Paranoid", "Master of Reality"]
    )
    assert formatted_filters[3] == dict(
        field="genre", clause="must", value=["Doom", "Heavy Metal"], operator="or"
    )

    formatted_filters = se.format_filters(filters, clause="must not")
    assert len(formatted_filters) == 4
    assert formatted_filters[0] == dict(
        field="year", clause="must not", value=1980, operator="gte"
    )
    assert formatted_filters[1] == dict(field="ozzy", clause="must not", value=True)
    assert formatted_filters[2] == dict(
        field="album", clause="must not", value=["Paranoid", "Master of Reality"]
    )
    assert formatted_filters[3] == dict(
        field="genre", clause="must not", value=["Doom", "Heavy Metal"], operator="or"
    )

    formatted_filters = se.format_filters({})
    assert formatted_filters == []


def test_search_filters_only(collection, schema):
    se = AdvancedRetriever(schema).index(collection)

    query = {
        "year": ("gte", 1970),
        "ozzy": True,
        "album": ["Paranoid", "Heaven and Hell"],
        "genre": ("or", ["Doom", "Heavy Metal"]),
    }

    res = se.search(query=query, return_docs=False)

    assert len(res) == 1
    assert res["doc_1"] == 1.0

    query = {
        "where": {
            "year": ("gt", 1969),
            "album": ["Paranoid", "Heaven and Hell"],
            "genre": ("or", ["Doom", "Heavy Metal"]),
        }
    }

    res = se.search(query=query, return_docs=False)

    assert len(res) == 2
    assert res["doc_1"] == 1.0
    assert res["doc_2"] == 1.0

    query = {
        "where_not": {
            "year": ("gt", 1969),
            "ozzy": False,
            "album": ["Paranoid", "Heaven and Hell"],
        }
    }

    res = se.search(query=query, return_docs=False)

    assert len(res) == 1
    assert res["doc_0"] == 1.0


def test_search_or(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    res = se.search(query="witches masses", return_docs=False)
    assert len(res) == 2
    assert "doc_0" in res
    assert "doc_1" in res


def test_search_and(collection, schema):
    se = AdvancedRetriever(schema).index(collection)
    res = se.search(query="witches masses", return_docs=False, operator="AND")
    assert len(res) == 1
    assert "doc_1" in res


def test_advanced_search(collection, schema):
    se = AdvancedRetriever(schema).index(collection)

    query = {
        "text": "witches masses",
        "year": ("gte", 1970),
        "ozzy": True,
        "album": ["Paranoid", "Heaven and Hell"],
        "genre": ("or", ["Doom", "Heavy Metal"]),
    }

    res = se.search(query=query, return_docs=False)
    assert len(res) == 1
    assert "doc_1" in res


def test_search_with_subset_doc_ids(collection, schema):
    se = AdvancedRetriever(schema).index(collection)

    res = se.search(
        query="witches masses", subset_doc_ids=["doc_1", "doc_2"], return_docs=False
    )
    assert len(res) == 1
    assert "doc_1" in res


def test_index_file(schema):
    se = AdvancedRetriever(schema).index_file(
        "tests/test_data/multifield_collection.jsonl"
    )

    query = {
        "text": "witches masses",
        "year": ("gte", 1970),
        "ozzy": True,
        "album": ["Paranoid", "Heaven and Hell"],
        "genre": ("or", ["Doom", "Heavy Metal"]),
    }

    res = se.search(query=query, return_docs=False)
    assert len(res) == 1
    assert "doc_1" in res


def test_load(schema):
    se = AdvancedRetriever.load("new-index")
    assert se.schema == schema
