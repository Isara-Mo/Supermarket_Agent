import streamlit as st
import pandas as pd
import os
from .config import SAVED_FILES_DIR, METADATA_FILE, DB_FILE
from .file_ops import (
    save_csv_file,
    save_pdf_files,
    load_saved_csv,
    check_saved_databases,
    load_metadata,
    save_metadata,
)
from .processing import pdf_read, get_chunks, process_product_csv, check_database_exists
from .embeddings import init_embeddings, init_llm
from .responses import get_pdf_response, get_supermarket_response, get_csv_response

st.set_page_config(
    page_title="智能超市个性化客服",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    :root { --primary-color: #1f77b4; --supermarket-color: #28a745; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .main-header { font-size: 3rem; text-align: center; margin-bottom: 2rem; }
    .info-card { background: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'pdf_messages' not in st.session_state:
        st.session_state.pdf_messages = []
    if 'csv_messages' not in st.session_state:
        st.session_state.csv_messages = []
    if 'supermarket_messages' not in st.session_state:
        st.session_state.supermarket_messages = []
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'product_df' not in st.session_state:
        st.session_state.product_df = None
    if 'current_supermarket_db' not in st.session_state:
        st.session_state.current_supermarket_db = None
    if 'current_pdf_db' not in st.session_state:
        st.session_state.current_pdf_db = None


def main():
    init_session_state()

    st.markdown('<h1 class="main-header">🤖 智能超市个性化客服</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem; color: #666;">集PDF问答、数据分析与超市客服于一体的智能助手</div>', unsafe_allow_html=True)

    saved_dbs = check_saved_databases()

    tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF智能问答", "📊 CSV数据分析", "🛒 超市智能客服", "📁 数据管理"])

    # PDF tab (simplified, calls into responses and processing)
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 💬 与PDF文档对话")
            for message in st.session_state.pdf_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"]) 

            can_chat = st.session_state.current_pdf_db and check_database_exists(st.session_state.current_pdf_db)
            if pdf_query := st.chat_input("💭 向PDF提问...", disabled=not can_chat):
                st.session_state.pdf_messages.append({"role": "user", "content": pdf_query})
                with st.chat_message("assistant"):
                    with st.spinner("🤔 AI正在分析文档..."):
                        response = get_pdf_response(pdf_query, st.session_state.current_pdf_db)
                    st.markdown(response)
                    st.session_state.pdf_messages.append({"role": "assistant", "content": response})

        with col2:
            st.markdown("### 📁 文档管理")
            pdf_dbs = [db for db in saved_dbs if db["file_type"] == "pdf"]
            if pdf_dbs:
                st.markdown("**📚 已保存的PDF数据库:**")
                for i, db_info in enumerate(pdf_dbs):
                    is_current = st.session_state.current_pdf_db == db_info["db_name"]
                    if st.button("选择", key=f"select_pdf_{i}", disabled=is_current):
                        st.session_state.current_pdf_db = db_info["db_name"]
                        st.session_state.pdf_messages = []
                        st.success("已切换PDF数据库")
                        st.rerun()

            pdf_docs = st.file_uploader("📎 上传PDF文件", accept_multiple_files=True, type=['pdf'], key="pdf_uploader")
            if pdf_docs:
                if st.button("🚀 上传并处理PDF文档", disabled=not pdf_docs, use_container_width=True):
                    with st.spinner("📊 正在处理PDF文件..."):
                        try:
                            metadata_key, db_name, saved_files = save_pdf_files(pdf_docs)
                            raw_text = pdf_read(pdf_docs)
                            if not raw_text.strip():
                                st.error("❌ 无法从PDF中提取文本")
                            else:
                                text_chunks = get_chunks(raw_text)
                                embeddings = init_embeddings()
                                # 创建向量数据库
                                from .processing import vector_store
                                vector_store(text_chunks, db_name, embeddings)
                                st.session_state.current_pdf_db = db_name
                                st.session_state.pdf_messages = []
                                st.success("✅ PDF处理完成！")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ 处理PDF时出错: {str(e)}")

    # CSV tab
    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📈 数据分析对话")
            for message in st.session_state.csv_messages:
                with st.chat_message(message["role"]):
                    if message.get("type") == "dataframe":
                        st.dataframe(message["content"])
                    else:
                        st.markdown(message["content"]) 

            if csv_query := st.chat_input("📊 分析数据...", disabled=st.session_state.df is None):
                st.session_state.csv_messages.append({"role": "user", "content": csv_query, "type": "text"})
                with st.chat_message("assistant"):
                    with st.spinner("🔄 正在分析数据..."):
                        response = get_csv_response(csv_query, st.session_state.df)
                    st.markdown(response)
                    st.session_state.csv_messages.append({"role": "assistant", "content": response, "type": "text"})

        with col2:
            st.markdown("### 📊 数据管理")
            csv_file = st.file_uploader("📈 上传CSV文件", type='csv', key="analysis_csv")
            if csv_file:
                st.session_state.df = pd.read_csv(csv_file)
                st.success("✅ 数据加载成功!")
                with st.expander("👀 数据预览", expanded=True):
                    st.dataframe(st.session_state.df.head())

    # 超市智能客服 tab
    with tab3:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🛒 超市智能客服")
            for message in st.session_state.supermarket_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"]) 

            current_db = st.session_state.current_supermarket_db
            can_chat = current_db and check_database_exists(current_db)
            if supermarket_query := st.chat_input("🛒 询问商品信息...", disabled=not can_chat):
                st.session_state.supermarket_messages.append({"role": "user", "content": supermarket_query})
                with st.chat_message("assistant"):
                    with st.spinner("🔍 正在查找商品信息..."):
                        response = get_supermarket_response(supermarket_query, current_db)
                    st.markdown(response)
                    st.session_state.supermarket_messages.append({"role": "assistant", "content": response})

        with col2:
            st.markdown("### 🏪 商品数据管理")
            product_csv = st.file_uploader("🛒 上传商品信息CSV", type='csv', key="product_csv")
            if product_csv:
                st.session_state.product_df = pd.read_csv(product_csv)
                st.success("✅ 商品数据加载成功!")
                with st.expander("👀 商品数据预览", expanded=True):
                    st.dataframe(st.session_state.product_df.head())

                if st.button("🚀 保存并处理商品数据", use_container_width=True):
                    with st.spinner("💾 正在保存文件和创建知识库..."):
                        try:
                            filename, file_path = save_csv_file(product_csv, "product")
                            metadata = load_metadata()
                            db_name = metadata[filename]["db_name"]
                            embeddings = init_embeddings()
                            success, result = process_product_csv(st.session_state.product_df, db_name, embeddings)
                            if success:
                                st.session_state.current_supermarket_db = db_name
                                st.session_state.supermarket_messages = []
                                st.success(f"✅ 商品数据已保存！创建了 {result} 个数据块")
                                st.rerun()
                            else:
                                st.error(f"❌ 处理失败: {result}")
                        except Exception as e:
                            st.error(f"❌ 保存或处理商品数据时出错: {str(e)}")

    # 数据管理 tab (simplified)
    with tab4:
        st.markdown("### 📁 数据管理中心")
        saved_dbs = check_saved_databases()
        if saved_dbs:
            for i, db_info in enumerate(saved_dbs):
                if st.button(f"选择: {db_info['original_name']}", key=f"choose_db_{i}"):
                    if db_info['file_type'] == 'product':
                        st.session_state.current_supermarket_db = db_info['db_name']
                        st.session_state.product_df = load_saved_csv(db_info['filename'])
                        st.success("已切换数据库")
                        st.rerun()
        else:
            st.info("🗂️ 暂无保存的数据文件")


if __name__ == "__main__":
    main()
