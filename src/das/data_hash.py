import hashlib
from pathlib import Path


def hash_data(data_path, chunk_size=65536):
    if Path(data_path).is_dir():
        hash = _hash_dir(data_path, chunk_size)
    elif Path(data_path).is_file():
        hash = _hash_file(data_path, chunk_size)
    else:
        raise ValueError(f"{data_path} is neither directory nor file.")
    return hash.hexdigest()


def _update_hash_dir(directory, hash, chunk_size):
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash.update(chunk)
        elif path.is_dir():
            hash = _update_hash_dir(path, hash, chunk_size)
    return hash


def _hash_dir(directory, chunk_size):
    return _update_hash_dir(directory, hashlib.md5(), chunk_size)


def _hash_file(data_file, chunk_size):
    hash = hashlib.md5()
    with open(data_file, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash.update(chunk)
    return hash