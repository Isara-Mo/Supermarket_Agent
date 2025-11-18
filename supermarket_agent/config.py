import os
from dotenv import load_dotenv

load_dotenv(override=True)

# API keys
DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
dashscope_api_key = os.getenv("dashscope_api_key")

# 防止某些本地 libraray 报错
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 目录和文件路径
SAVED_FILES_DIR = "./data_backup/saved_files"
METADATA_FILE = "./data_backup/file_metadata.json"
DB_FILE = "./data_backup/db"

os.makedirs(SAVED_FILES_DIR, exist_ok=True)
os.makedirs(DB_FILE, exist_ok=True)
