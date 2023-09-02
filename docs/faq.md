# FAQ

## How do I change `retriv` working directory?
```python
import retriv

retriv.set_base_path("new/working/path")
```

or

```python
import os

os.environ["RETRIV_BASE_PATH"] = "new/working/path"
```
