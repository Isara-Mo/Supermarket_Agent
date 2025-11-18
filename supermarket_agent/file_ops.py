import os
import json
import hashlib
from datetime import datetime
import pandas as pd

from .config import SAVED_FILES_DIR, METADATA_FILE, DB_FILE


def get_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_metadata(metadata: dict):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_csv_file(uploaded_file, file_type: str = "product"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{file_type}_{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(SAVED_FILES_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_hash = get_file_hash(file_path)

    metadata = load_metadata()
    metadata[filename] = {
        "original_name": uploaded_file.name,
        "file_type": file_type,
        "upload_time": timestamp,
        "file_hash": file_hash,
        "file_path": file_path,
        "db_name": f"{file_type}_db_{timestamp}"
    }
    save_metadata(metadata)

    return filename, file_path


def save_pdf_files(uploaded_files):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    for uploaded_file in uploaded_files:
        filename = f"pdf_{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(SAVED_FILES_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(filename)

    combined_hash = hashlib.md5()
    for filename in saved_files:
        file_path = os.path.join(SAVED_FILES_DIR, filename)
        combined_hash.update(get_file_hash(file_path).encode())

    metadata = load_metadata()
    db_name = f"pdf_db_{timestamp}"
    metadata_key = f"pdf_group_{timestamp}"

    metadata[metadata_key] = {
        "original_name": [f.name for f in uploaded_files],
        "saved_files": saved_files,
        "file_type": "pdf",
        "upload_time": timestamp,
        "file_hash": combined_hash.hexdigest(),
        "db_name": db_name
    }
    save_metadata(metadata)

    return metadata_key, db_name, saved_files


def load_saved_csv(filename: str):
    metadata = load_metadata()
    if filename in metadata:
        file_path = metadata[filename].get("file_path")
        if file_path and os.path.exists(file_path):
            return pd.read_csv(file_path)
    return None


def check_saved_databases() -> list:
    saved_dbs = []
    metadata = load_metadata()
    for filename, info in metadata.items():
        db_name = info.get("db_name", "")
        db_path = os.path.join(DB_FILE, db_name)
        if os.path.exists(db_path) and os.path.exists(f"{db_path}/index.faiss"):
            saved_dbs.append({
                "filename": filename,
                "original_name": info.get("original_name", "Unknown"),
                "upload_time": info.get("upload_time", ""),
                "file_type": info.get("file_type", ""),
                "db_name": db_name,
                "file_path": info.get("file_path", ""),
                "saved_files": info.get("saved_files", [])
            })
    return saved_dbs
