"""
Hashing utilities for Clarifai Python SDK.

This module provides functions for computing stable hashes of directories and files,
commonly used for change detection in pipeline steps and other components.
"""

import hashlib
import os
from typing import List, Optional


def hash_directory(
    directory: str, algo: str = "md5", exclude_files: Optional[List[str]] = None
) -> str:
    """
    Compute a stable hash of all files in a directory.

    This function computes a hash that accounts for:
    - File relative paths (to detect renames)
    - File sizes (to detect empty files)
    - File contents (read in chunks for large files)

    :param directory: Directory to hash
    :param algo: Hash algorithm ('md5', 'sha1', 'sha256', etc.)
    :param exclude_files: List of file names to exclude from hash calculation.
                         If None, defaults to ['config-lock.yaml'] for backward compatibility.
    :return: Hash as lowercase hex digest string
    """
    if exclude_files is None:
        exclude_files = ['config-lock.yaml']

    # Ensure directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")

    hash_func = hashlib.new(algo)

    for root, _, files in os.walk(directory):
        for name in sorted(files):
            # Skip files in the exclusion list
            if name in exclude_files:
                continue

            filepath = os.path.join(root, name)
            relative_path = os.path.relpath(filepath, directory)

            # Hash the relative path to detect renames
            hash_func.update(relative_path.encode("utf-8"))

            # Hash the file size to detect empties
            file_size = os.path.getsize(filepath)
            hash_func.update(str(file_size).encode("utf-8"))

            # Hash the file contents (read in chunks for large files)
            try:
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_func.update(chunk)
            except (IOError, OSError) as e:
                # If we can't read the file, include the error in the hash
                # This ensures the hash changes if file permissions change
                hash_func.update(f"ERROR_READING_FILE: {e}".encode("utf-8"))

    return hash_func.hexdigest()


def hash_file(filepath: str, algo: str = "md5") -> str:
    """
    Compute a hash of a single file.

    :param filepath: Path to the file to hash
    :param algo: Hash algorithm ('md5', 'sha1', 'sha256', etc.)
    :return: Hash as lowercase hex digest string
    """
    if not os.path.exists(filepath):
        raise ValueError(f"File does not exist: {filepath}")

    if not os.path.isfile(filepath):
        raise ValueError(f"Path is not a file: {filepath}")

    hash_func = hashlib.new(algo)

    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
    except (IOError, OSError) as e:
        raise ValueError(f"Error reading file {filepath}: {e}")

    return hash_func.hexdigest()


def verify_hash_algorithm(algo: str) -> bool:
    """
    Verify that a hash algorithm is supported.

    :param algo: Hash algorithm name
    :return: True if algorithm is supported, False otherwise
    """
    try:
        hashlib.new(algo)
        return True
    except ValueError:
        return False


def get_available_algorithms() -> List[str]:
    """
    Get list of available hash algorithms.

    :return: List of available algorithm names
    """
    return list(hashlib.algorithms_available)
