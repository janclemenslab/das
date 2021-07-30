import hashlib
from pathlib import Path
from typing import Union


def hash_data(data_path: Union[str, Path], chunk_size: int = 65536):
    """Compute MD5 hash of the data_path (dir or file) for data versioning.

    Args:
        data_path ([type]): [description]
        chunk_size (int, optional): [description]. Defaults to 65536.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if Path(data_path).is_dir():
        hash = _hash_dir(data_path, chunk_size)
    elif Path(data_path).is_file():
        hash = _hash_file(data_path, chunk_size)
    else:
        raise ValueError(f"{data_path} is neither directory nor file.")
    return hash.hexdigest()


def _update_hash_dir(directory: Union[str, Path], hash, chunk_size: int):
    # from https://stackoverflow.com/questions/24937495/how-can-i-calculate-a-hash-for-a-filesystem-directory-using-python/54477583#54477583
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


def _hash_dir(directory: Union[str, Path], chunk_size: int):
    return _update_hash_dir(directory, hashlib.md5(), chunk_size)


def _hash_file(data_file: Union[str, Path], chunk_size: int):
    hash = hashlib.md5()
    with open(data_file, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash.update(chunk)
    return hash