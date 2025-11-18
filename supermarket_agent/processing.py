from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .config import DB_FILE


def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks: list, db_name: str, embeddings):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    db_path = __import__("os").path.join(DB_FILE, db_name)
    vector_store.save_local(db_path)


def csv_to_text(df):
    text_chunks = []
    for index, row in df.iterrows():
        row_text = f"商品信息 {index + 1}:\n"
        for column, value in row.items():
            row_text += f"{column}: {value}\n"
        row_text += "\n"
        text_chunks.append(row_text)
    return text_chunks


def process_product_csv(df, db_name, embeddings):
    try:
        text_chunks = csv_to_text(df)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        final_chunks = []
        for chunk in text_chunks:
            final_chunks.extend(text_splitter.split_text(chunk))

        vector_store(final_chunks, db_name, embeddings)
        return True, len(final_chunks)
    except Exception as e:
        return False, str(e)


def check_database_exists(db_name: str, db_folder: str = DB_FILE) -> bool:
    if not db_name:
        return False
    import os
    db_path = os.path.join(db_folder, db_name)
    return os.path.exists(db_path) and os.path.exists(f"{db_path}/index.faiss")
