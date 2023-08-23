import os
from pathlib import Path


def base_path():
    p = Path(os.environ.get("RETRIV_BASE_PATH"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def collections_path():
    p = base_path() / "collections"
    p.mkdir(parents=True, exist_ok=True)
    return p


def index_path(index_name: str):
    path = collections_path() / index_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def docs_path(index_name: str):
    return index_path(index_name) / "docs.jsonl"


def sr_state_path(index_name: str):
    return index_path(index_name) / "sr_state.npz"


def fr_state_path(index_name: str):
    return index_path(index_name) / "fr_state.npz"


def embeddings_path(index_name: str):
    return index_path(index_name) / "embeddings.h5"


def embeddings_folder_path(index_name: str):
    path = index_path(index_name) / "embeddings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def faiss_index_path(index_name: str):
    return index_path(index_name) / "faiss.index"


def faiss_index_infos_path(index_name: str):
    return index_path(index_name) / "faiss_index_infos.json"


def dr_state_path(index_name: str):
    return index_path(index_name) / "dr_state.npz"


def hr_state_path(index_name: str):
    return index_path(index_name) / "hr_state.npz"


def encoder_state_path(index_name: str):
    return index_path(index_name) / "encoder_state.json"


def merger_state_path(index_name: str):
    return index_path(index_name) / "merger_state.npz"
