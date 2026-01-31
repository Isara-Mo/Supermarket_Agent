import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 目录配置
SAVED_FILES_DIR = "./data_backup/saved_files"
METADATA_FILE = "./data_backup/file_metadata.json"
DB_FILE = "./data_backup/db"

def init_dirs():
    os.makedirs(SAVED_FILES_DIR, exist_ok=True)
    os.makedirs(DB_FILE, exist_ok=True)

def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f: json.dump(metadata, f, ensure_ascii=False, indent=2)

def save_csv_file(uploaded_file, file_type="product"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{file_type}_{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(SAVED_FILES_DIR, filename)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    metadata = load_metadata()
    metadata[filename] = {
        "original_name": uploaded_file.name, "file_type": file_type,
        "upload_time": timestamp, "file_path": file_path,
        "db_name": f"{file_type}_db_{timestamp}"
    }
    save_metadata(metadata)
    return filename, file_path

def load_saved_csv(filename):
    metadata = load_metadata()
    if filename in metadata:
        path = metadata[filename]["file_path"]
        if os.path.exists(path): return pd.read_csv(path)
    return None

def check_saved_databases():
    saved_dbs = []
    metadata = load_metadata()
    for filename, info in metadata.items():
        db_path = os.path.join(DB_FILE, info.get("db_name", ""))
        if os.path.exists(db_path) and os.path.exists(f"{db_path}/index.faiss"):
            db_info = info.copy()
            db_info["filename"] = filename
            saved_dbs.append(db_info)
    return saved_dbs

def check_database_exists(db_name="faiss_db"):
    path = os.path.join(DB_FILE, db_name)
    return os.path.exists(path) and os.path.exists(f"{path}/index.faiss")

def process_product_csv(df, db_name, embeddings_func):
    """处理CSV并保存到向量库"""
    try:
        # 将整行转换为文本
        text_data = df.astype(str).apply(lambda x: " | ".join(x), axis=1).tolist()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents(text_data)
        
        vectorstore = FAISS.from_documents(docs, embeddings_func)
        vectorstore.save_local(os.path.join(DB_FILE, db_name))
        return True, len(docs)
    except Exception as e:
        return False, str(e)