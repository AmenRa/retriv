## Filtering Search Results

[retriv](https://github.com/AmenRa/retriv) supports two way of filtering the search results (`where` and `where_not`) and several type-specific operators.

- `where` means that only the documents matching the filter will be considered during search.
- `where_not` means that the documents matching the filter will be ignored during search.

Below we describe the effects of the supported operators for each data type and way of filtering.

### Where

| Field Type | Operator  | Value                      | Meaning                                                                                                                                              |
| ---------- | --------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| number     | `eq`      | number                     | Only the documents whose field value is **equal to** the provided value will be considered during search.                                            |
| number     | `gt`      | number                     | Only the documents whose field value is **greater than** the provided value will be considered during search.                                        |
| number     | `gte`     | number                     | Only the documents whose field value is **greater or equal to** the provided value will be considered during search.                                 |
| number     | `lt`      | number                     | Only the documents whose field value is **less than** the provided value will be considered during search.                                           |
| number     | `lte`     | number                     | Only the documents whose field value is **less or equal to** the provided value will be considered during search.                                    |
| number     | `between` | number                     | Only the documents whose field value is **between** the provided values (included) will be considered during search.                                 |
| bool       |           | True / False               | Only the documents whose field value is **equal to** the provided value will be considered during search.                                            |
| keyword    |           | any value / list of values | Only the documents whose field value is **equal to** the provided value or **among** the provided values will be considered during search.           |
| keywords   | `or`      | any value / list of values | Only the documents whose field value is **contains** the provided value or **contains one of** the provided values will be considered during search. |
| keywords   | `and`     | any value / list of values | Only the documents whose field value **contains all** the provided values will be considered during search.                                          |

Query example:
```python
query = {
    "text": "search terms",
    "where": {
        "numeric_field_name": ("gte", 1970),
        "boolean_field_name": True,
        "keyword_field_name": "kw_1",
        "keywords_field_name": ("or", ["kws_23", "kws_666"]),
    }
}
```

Alternatively, you can omit the `where` key and use the following syntax:
```python
query = {
    "text": "search terms",
    "numeric_field_name": ("gte", 1970),
    "boolean_field_name": True,
    "keyword_field_name": "kw_1",
    "keywords_field_name": ("or", ["kws_23", "kws_666"]),
}
```


### Where not

| Field Type | Operator  | Value                      | Meaning                                                                                                                        |
| ---------- | --------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| number     | `eq`      | number                     | The documents whose field value is **equal to** the provided value will be ignored.                                            |
| number     | `gt`      | number                     | The documents whose field value is **greater than** the provided value will be ignored.                                        |
| number     | `gte`     | number                     | The documents whose field value is **greater or equal to** the provided value will be ignored.                                 |
| number     | `lt`      | number                     | The documents whose field value is **less than** the provided value will be ignored.                                           |
| number     | `lte`     | number                     | The documents whose field value is **less or equal to** the provided value will be ignored.                                    |
| number     | `between` | number                     | The documents whose field value is **between** the provided values (included) will be ignored.                                 |
| bool       |           | True / False               | The documents whose field value is **equal to** the provided value will be ignored.                                            |
| keyword    |           | any value / list of values | The documents whose field value is **equal to** the provided value or **among** the provided values will be ignored.           |
| keywords   | `or`      | any value / list of values | The documents whose field value is **contains** the provided value or **contains one of** the provided values will be ignored. |
| keywords   | `and`     | any value / list of values | The documents whose field value **contains all** the provided values will be ignored.                                          |

Query example:
```python
query = {
    "text": "search terms",
    "where": {
        "numeric_field_name": ("gte", 1970),
        "boolean_field_name": True,
        "keyword_field_name": "kw_1",
        "keywords_field_name": ("or", ["kws_23", "kws_666"]),
    }
}
```