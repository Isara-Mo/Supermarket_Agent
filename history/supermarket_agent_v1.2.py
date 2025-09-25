import streamlit as st
import pandas as pd
import os
import json
import hashlib
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
from langchain_experimental.tools import PythonAstREPLTool
import matplotlib
matplotlib.use('Agg')
import os
from dotenv import load_dotenv 
load_dotenv(override=True)

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
dashscope_api_key = os.getenv("dashscope_api_key")

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 创建必要的目录
SAVED_FILES_DIR = "saved_files"
METADATA_FILE = "file_metadata.json"
os.makedirs(SAVED_FILES_DIR, exist_ok=True)

# 页面配置
st.set_page_config(
    page_title="智能超市个性化客服",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [保持原有的CSS样式不变]
st.markdown("""
<style>
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff9800;
        --error-color: #d62728;
        --background-color: #f8f9fa;
        --supermarket-color: #28a745;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #28a745);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .success-card {
        background: linear-gradient(135deg, #e8f5e8, #f0f8f0);
        border-left: 4px solid var(--success-color);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff8e1, #fffbf0);
        border-left: 4px solid var(--warning-color);
    }
    
    .supermarket-card {
        background: linear-gradient(135deg, #e8f8f0, #f0fff4);
        border-left: 4px solid var(--supermarket-color);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #1f77b4, #2196F3);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }
    
    .file-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--supermarket-color);
    }
</style>
""", unsafe_allow_html=True)

# 新增：文件管理功能
def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_metadata():
    """加载文件元数据"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    """保存文件元数据"""
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def save_csv_file(uploaded_file, file_type="product"):
    """保存上传的CSV文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{file_type}_{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(SAVED_FILES_DIR, filename)
    
    # 保存文件
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 计算哈希值
    file_hash = get_file_hash(file_path)
    
    # 更新元数据
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

def load_saved_csv(filename):
    """加载已保存的CSV文件"""
    metadata = load_metadata()
    if filename in metadata:
        file_path = metadata[filename]["file_path"]
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
    return None

def check_saved_databases():
    """检查已保存的数据库"""
    saved_dbs = []
    metadata = load_metadata()
    
    for filename, info in metadata.items():
        db_name = info.get("db_name", "")
        if os.path.exists(db_name) and os.path.exists(f"{db_name}/index.faiss"):
            saved_dbs.append({
                "filename": filename,
                "original_name": info["original_name"],
                "upload_time": info["upload_time"],
                "file_type": info["file_type"],
                "db_name": db_name,
                "file_path": info["file_path"]
            })
    
    return saved_dbs

# 初始化embeddings
@st.cache_resource
def init_embeddings():
    return DashScopeEmbeddings(
        model="text-embedding-v1", 
        dashscope_api_key=dashscope_api_key
    )

# 初始化LLM
@st.cache_resource
def init_llm():
    return init_chat_model("deepseek-chat", model_provider="deepseek")

# 初始化会话状态
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

# [保持原有的PDF处理函数不变]
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks, db_name="faiss_db"):
    embeddings = init_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(db_name)

def check_database_exists(db_name="faiss_db"):
    return os.path.exists(db_name) and os.path.exists(f"{db_name}/index.faiss")

def get_pdf_response(user_question):
    if not check_database_exists("faiss_db"):
        return "❌ 请先上传PDF文件并点击'Submit & Process'按钮来处理文档！"
    
    try:
        embeddings = init_embeddings()
        llm = init_llm()
        
        new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是AI助手，请根据提供的上下文回答问题，确保提供所有细节，如果答案不在上下文中，请说"答案不在上下文中"，不要提供错误的答案"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf")
        agent = create_tool_calling_agent(llm, [retrieval_chain], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[retrieval_chain], verbose=True)
        
        response = agent_executor.invoke({"input": user_question})
        return response['output']
        
    except Exception as e:
        return f"❌ 处理问题时出错: {str(e)}"

# CSV数据处理为文本
def csv_to_text(df):
    """将CSV数据转换为文本格式用于RAG"""
    text_chunks = []
    
    for index, row in df.iterrows():
        row_text = f"商品信息 {index + 1}:\n"
        for column, value in row.items():
            row_text += f"{column}: {value}\n"
        row_text += "\n"
        text_chunks.append(row_text)
    
    return text_chunks

def process_product_csv(df, db_name):
    """处理商品CSV数据并创建向量数据库"""
    try:
        text_chunks = csv_to_text(df)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        final_chunks = []
        for chunk in text_chunks:
            final_chunks.extend(text_splitter.split_text(chunk))
        
        vector_store(final_chunks, db_name)
        return True, len(final_chunks)
    except Exception as e:
        return False, str(e)

def get_supermarket_response(user_question, db_name="supermarket_db"):
    """处理超市客服问题"""
    if not check_database_exists(db_name):
        return "❌ 请先选择或上传商品CSV文件！"
    
    try:
        embeddings = init_embeddings()
        llm = init_llm()
        
        new_db = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever(search_kwargs={"k": 5})
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个超市的智能客服助手。请根据提供的商品信息回答顾客的问题。

你的职责包括：
1. 帮助顾客查找商品信息（价格、库存、规格等）
2. 推荐相关商品
3. 回答关于商品的各种问题
4. 提供购物建议

回答时请：
- 友好热情，就像真正的超市客服
- 基于提供的商品数据给出准确信息
- 如果没有相关商品信息，诚实告知并建议其他方案
- 适当使用表情符号让对话更生动

如果顾客询问的商品不在数据库中，请说"很抱歉，我没有找到相关商品信息，请您到店内咨询工作人员或者尝试描述更具体的商品信息。" """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        retrieval_chain = create_retriever_tool(
            retriever, 
            "product_search", 
            "This tool searches for product information in the supermarket database"
        )
        agent = create_tool_calling_agent(llm, [retrieval_chain], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[retrieval_chain], verbose=True)
        
        response = agent_executor.invoke({"input": user_question})
        return response['output']
        
    except Exception as e:
        return f"❌ 处理问题时出错: {str(e)}"

# [保持原有的CSV数据分析函数不变]
def get_csv_response(query: str) -> str:
    if st.session_state.df is None:
        return "请先上传CSV文件"
    
    llm = init_llm()
    locals_dict = {'df': st.session_state.df}
    tools = [PythonAstREPLTool(locals=locals_dict)]
    
    system = f"""Given a pandas dataframe `df` answer user's query.
    Here's the output of `df.head().to_markdown()` for your reference, you have access to full dataframe as `df`:
    ```
    {st.session_state.df.head().to_markdown()}
    ```
    Give final answer as soon as you have enough data, otherwise generate code using `df` and call required tool.
    If user asks you to make a graph, save it as `plot.png`, and output GRAPH:<graph title>.
    Example:
    ```
    plt.hist(df['Age'])
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age Histogram')
    plt.savefig('plot.png')
    ``` output: GRAPH:Age histogram
    Query:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor.invoke({"input": query})['output']

def main():
    init_session_state()
    
    # 主标题
    st.markdown('<h1 class="main-header">🤖 智能超市个性化客服</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem; color: #666;">集PDF问答、数据分析与超市客服于一体的智能助手</div>', unsafe_allow_html=True)
    
    # 检查已保存的数据库
    saved_dbs = check_saved_databases()
    
    # 创建三个主要功能的标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF智能问答", "📊 CSV数据分析", "🛒 超市智能客服", "📁 数据管理"])
    
    # [PDF问答模块保持不变]
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💬 与PDF文档对话")
            
            if check_database_exists("faiss_db"):
                st.markdown('<div class="info-card success-card"><span class="status-indicator status-ready">✅ PDF数据库已准备就绪</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card warning-card"><span class="status-indicator status-waiting">⚠️ 请先上传并处理PDF文件</span></div>', unsafe_allow_html=True)
            
            for message in st.session_state.pdf_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if pdf_query := st.chat_input("💭 向PDF提问...", disabled=not check_database_exists("faiss_db")):
                st.session_state.pdf_messages.append({"role": "user", "content": pdf_query})
                with st.chat_message("user"):
                    st.markdown(pdf_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 AI正在分析文档..."):
                        response = get_pdf_response(pdf_query)
                    st.markdown(response)
                    st.session_state.pdf_messages.append({"role": "assistant", "content": response})
        
        with col2:
            st.markdown("### 📁 文档管理")
            
            pdf_docs = st.file_uploader(
                "📎 上传PDF文件",
                accept_multiple_files=True,
                type=['pdf'],
                help="支持上传多个PDF文件",
                key="pdf_uploader"
            )
            
            if pdf_docs:
                st.success(f"📄 已选择 {len(pdf_docs)} 个文件")
                for i, pdf in enumerate(pdf_docs, 1):
                    st.write(f"• {pdf.name}")
            
            if st.button("🚀 上传并处理PDF文档", disabled=not pdf_docs, use_container_width=True):
                with st.spinner("📊 正在处理PDF文件..."):
                    try:
                        raw_text = pdf_read(pdf_docs)
                        if not raw_text.strip():
                            st.error("❌ 无法从PDF中提取文本")
                            return
                        
                        text_chunks = get_chunks(raw_text)
                        st.info(f"📝 文本已分割为 {len(text_chunks)} 个片段")
                        
                        vector_store(text_chunks, "faiss_db")
                        st.success("✅ PDF处理完成！")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ 处理PDF时出错: {str(e)}")
            
            if st.button("🗑️ 清除PDF数据库", use_container_width=True):
                try:
                    import shutil
                    if os.path.exists("faiss_db"):
                        shutil.rmtree("faiss_db")
                    st.session_state.pdf_messages = []
                    st.success("数据库已清除")
                    st.rerun()
                except Exception as e:
                    st.error(f"清除失败: {e}")
    
    # [CSV数据分析模块保持不变]
    with tab2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 数据分析对话")
            
            if st.session_state.df is not None:
                st.markdown('<div class="info-card success-card"><span class="status-indicator status-ready">✅ 数据已加载完成</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card warning-card"><span class="status-indicator status-waiting">⚠️ 请先上传CSV文件</span></div>', unsafe_allow_html=True)
            
            for message in st.session_state.csv_messages:
                with st.chat_message(message["role"]):
                    if message["type"] == "dataframe":
                        st.dataframe(message["content"])
                    elif message["type"] == "image":
                        st.write(message["content"])
                        if os.path.exists('plot.png'):
                            st.image('plot.png')
                    else:
                        st.markdown(message["content"])
            
            if csv_query := st.chat_input("📊 分析数据...", disabled=st.session_state.df is None):
                st.session_state.csv_messages.append({"role": "user", "content": csv_query, "type": "text"})
                with st.chat_message("user"):
                    st.markdown(csv_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("🔄 正在分析数据..."):
                        response = get_csv_response(csv_query)
                    
                    if isinstance(response, pd.DataFrame):
                        st.dataframe(response)
                        st.session_state.csv_messages.append({"role": "assistant", "content": response, "type": "dataframe"})
                    elif "GRAPH" in str(response):
                        text = str(response)[str(response).find("GRAPH")+6:]
                        st.write(text)
                        if os.path.exists('plot.png'):
                            st.image('plot.png')
                        st.session_state.csv_messages.append({"role": "assistant", "content": text, "type": "image"})
                    else:
                        st.markdown(response)
                        st.session_state.csv_messages.append({"role": "assistant", "content": response, "type": "text"})
        
        with col2:
            st.markdown("### 📊 数据管理")
            
            csv_file = st.file_uploader("📈 上传CSV文件", type='csv', key="analysis_csv")
            if csv_file:
                st.session_state.df = pd.read_csv(csv_file)
                st.success(f"✅ 数据加载成功!")
                
                with st.expander("👀 数据预览", expanded=True):
                    st.dataframe(st.session_state.df.head())
                    st.write(f"📏 数据维度: {st.session_state.df.shape[0]} 行 × {st.session_state.df.shape[1]} 列")
            
            if st.session_state.df is not None:
                if st.button("📋 显示数据信息", use_container_width=True):
                    with st.expander("📊 数据统计信息", expanded=True):
                        st.write("**基本信息:**")
                        st.text(f"行数: {st.session_state.df.shape[0]}")
                        st.text(f"列数: {st.session_state.df.shape[1]}")
                        st.write("**列名:**")
                        st.write(list(st.session_state.df.columns))
                        st.write("**数据类型:**")
                        dtype_info = pd.DataFrame({
                            '列名': st.session_state.df.columns,
                            '数据类型': [str(dtype) for dtype in st.session_state.df.dtypes]
                        })
                        st.dataframe(dtype_info, use_container_width=True)
            
            if st.button("🗑️ 清除CSV数据", use_container_width=True, key="clear_csv"):
                st.session_state.df = None
                st.session_state.csv_messages = []
                if os.path.exists('plot.png'):
                    os.remove('plot.png')
                st.success("数据已清除")
                st.rerun()
    
    # 改进的超市智能客服模块
    with tab3:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🛒 超市智能客服")
            
            # 显示当前使用的数据库状态
            if st.session_state.current_supermarket_db and check_database_exists(st.session_state.current_supermarket_db):
                st.markdown(f'<div class="info-card supermarket-card"><span class="status-indicator status-supermarket">🛒 当前使用数据库: {st.session_state.current_supermarket_db}</span></div>', unsafe_allow_html=True)
            elif saved_dbs:
                st.markdown('<div class="info-card warning-card"><span class="status-indicator status-waiting">⚠️ 请在右侧选择一个已保存的数据库或上传新文件</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card warning-card"><span class="status-indicator status-waiting">⚠️ 请先上传商品信息CSV文件</span></div>', unsafe_allow_html=True)
            
            # 欢迎消息
            current_db = st.session_state.current_supermarket_db
            if not st.session_state.supermarket_messages and current_db and check_database_exists(current_db):
                welcome_msg = "🛒 欢迎来到智能超市！我是您的专属客服助手，可以帮您：\n\n• 🔍 查找商品信息\n• 💰 了解价格详情\n• 📦 查询库存状态\n• 🎯 推荐相关商品\n• 💡 提供购物建议\n\n请问今天需要什么帮助呢？"
                st.session_state.supermarket_messages.append({"role": "assistant", "content": welcome_msg})
            
            # 聊天界面
            for message in st.session_state.supermarket_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 用户输入
            can_chat = current_db and check_database_exists(current_db)
            if supermarket_query := st.chat_input("🛒 询问商品信息...", disabled=not can_chat):
                st.session_state.supermarket_messages.append({"role": "user", "content": supermarket_query})
                with st.chat_message("user"):
                    st.markdown(supermarket_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("🔍 正在查找商品信息..."):
                        response = get_supermarket_response(supermarket_query, current_db)
                    st.markdown(response)
                    st.session_state.supermarket_messages.append({"role": "assistant", "content": response})
        
        with col2:
            st.markdown("### 🏪 商品数据管理")
            
            # 显示已保存的数据库
            if saved_dbs:
                st.markdown("**📚 已保存的商品数据库:**")
                for i, db_info in enumerate(saved_dbs):
                    if db_info["file_type"] == "product":
                        is_current = st.session_state.current_supermarket_db == db_info["db_name"]
                        status_icon = "🟢" if is_current else "⚪"
                        
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"{status_icon} **{db_info['original_name']}**")
                            st.caption(f"上传时间: {db_info['upload_time']}")
                        with col_b:
                            if st.button("选择", key=f"select_{i}", disabled=is_current, use_container_width=True):
                                st.session_state.current_supermarket_db = db_info["db_name"]
                                st.session_state.product_df = load_saved_csv(db_info["filename"])
                                st.session_state.supermarket_messages = []  # 清除之前的对话
                                st.success(f"已切换到: {db_info['original_name']}")
                                st.rerun()
                
                st.markdown("---")
            
            # 上传新的商品CSV文件
            st.markdown("**📤 上传新的商品数据:**")
            product_csv = st.file_uploader(
                "🛒 上传商品信息CSV", 
                type='csv', 
                key="product_csv",
                help="上传包含商品名称、价格、类别、库存等信息的CSV文件"
            )
            
            if product_csv:
                st.session_state.product_df = pd.read_csv(product_csv)
                st.success(f"✅ 商品数据加载成功!")
                
                with st.expander("👀 商品数据预览", expanded=True):
                    st.dataframe(st.session_state.product_df.head())
                    st.write(f"📏 商品数据: {st.session_state.product_df.shape[0]} 种商品 × {st.session_state.product_df.shape[1]} 个字段")
                    
                st.write("**数据字段:**")
                for col in st.session_state.product_df.columns:
                    st.write(f"• {col}")
            
            # 处理商品数据
            if st.session_state.product_df is not None and product_csv:
                if st.button("🚀 保存并处理商品数据", use_container_width=True):
                    with st.spinner("💾 正在保存文件和创建知识库..."):
                        try:
                            # 保存文件
                            filename, file_path = save_csv_file(product_csv, "product")
                            
                            # 获取数据库名称
                            metadata = load_metadata()
                            db_name = metadata[filename]["db_name"]
                            
                            # 处理数据并创建向量数据库
                            success, result = process_product_csv(st.session_state.product_df, db_name)
                            
                            if success:
                                st.session_state.current_supermarket_db = db_name
                                st.session_state.supermarket_messages = []  # 清除之前的对话
                                st.success(f"✅ 商品数据已保存！创建了 {result} 个数据块")
                                st.info(f"📁 文件已保存为: {filename}")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ 处理失败: {result}")
                                
                        except Exception as e:
                            st.error(f"❌ 保存或处理商品数据时出错: {str(e)}")
            
            # 当前数据库的统计信息
            if st.session_state.product_df is not None:
                with st.expander("📊 商品统计", expanded=False):
                    st.write(f"**商品总数:** {len(st.session_state.product_df)}")
                    if '类别' in st.session_state.product_df.columns or 'category' in st.session_state.product_df.columns:
                        category_col = '类别' if '类别' in st.session_state.product_df.columns else 'category'
                        st.write("**商品分类统计:**")
                        category_counts = st.session_state.product_df[category_col].value_counts()
                        for category, count in category_counts.head().items():
                            st.write(f"• {category}: {count}种")
            
            # 示例问题
            current_db = st.session_state.current_supermarket_db
            if current_db and check_database_exists(current_db):
                st.markdown("### 💡 试试这些问题")
                example_questions = [
                    "有什么特价商品吗？",
                    "推荐一些水果",
                    "面包的价格是多少？",
                    "有机食品有哪些？",
                    "库存最多的商品是什么？"
                ]
                
                for question in example_questions:
                    if st.button(f"💭 {question}", key=f"example_{question}", use_container_width=True):
                        st.session_state.supermarket_messages.append({"role": "user", "content": question})
                        
                        with st.spinner("🔍 正在查找商品信息..."):
                            response = get_supermarket_response(question, current_db)
                        
                        st.session_state.supermarket_messages.append({"role": "assistant", "content": response})
                        st.rerun()
    
    # 新增：数据管理标签页
    with tab4:
        st.markdown("### 📁 数据管理中心")
        
        # 显示所有已保存的文件
        if saved_dbs:
            st.markdown("#### 📚 已保存的数据文件")
            
            for i, db_info in enumerate(saved_dbs):
                with st.expander(f"📄 {db_info['original_name']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**文件信息:**")
                        st.write(f"• 原文件名: {db_info['original_name']}")
                        st.write(f"• 文件类型: {db_info['file_type']}")
                        st.write(f"• 上传时间: {db_info['upload_time']}")
                        st.write(f"• 数据库名: {db_info['db_name']}")
                    
                    with col2:
                        st.write("**操作:**")
                        
                        # 预览数据
                        if st.button("👀 预览数据", key=f"preview_{i}"):
                            try:
                                df = load_saved_csv(db_info['filename'])
                                if df is not None:
                                    st.dataframe(df.head())
                                    st.write(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
                                else:
                                    st.error("无法加载数据")
                            except Exception as e:
                                st.error(f"预览失败: {str(e)}")
                        
                        # 删除文件和数据库
                        if st.button("🗑️ 删除", key=f"delete_{i}", type="secondary"):
                            try:
                                import shutil
                                # 删除数据库目录
                                if os.path.exists(db_info['db_name']):
                                    shutil.rmtree(db_info['db_name'])
                                
                                # 删除原文件
                                if os.path.exists(db_info['file_path']):
                                    os.remove(db_info['file_path'])
                                
                                # 更新元数据
                                metadata = load_metadata()
                                if db_info['filename'] in metadata:
                                    del metadata[db_info['filename']]
                                    save_metadata(metadata)
                                
                                # 如果删除的是当前使用的数据库，清除状态
                                if st.session_state.current_supermarket_db == db_info['db_name']:
                                    st.session_state.current_supermarket_db = None
                                    st.session_state.product_df = None
                                    st.session_state.supermarket_messages = []
                                
                                st.success(f"已删除: {db_info['original_name']}")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"删除失败: {str(e)}")
        else:
            st.info("🗂️ 暂无保存的数据文件")
        
        # 系统清理
        st.markdown("---")
        st.markdown("#### 🧹 系统清理")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清除所有PDF数据", use_container_width=True):
                try:
                    import shutil
                    if os.path.exists("faiss_db"):
                        shutil.rmtree("faiss_db")
                    st.session_state.pdf_messages = []
                    st.success("所有PDF数据已清除")
                    st.rerun()
                except Exception as e:
                    st.error(f"清除失败: {e}")
        
        with col2:
            if st.button("🗑️ 清除所有商品数据", use_container_width=True):
                try:
                    import shutil
                    # 删除所有商品数据库
                    for db_info in saved_dbs:
                        if os.path.exists(db_info['db_name']):
                            shutil.rmtree(db_info['db_name'])
                        if os.path.exists(db_info['file_path']):
                            os.remove(db_info['file_path'])
                    
                    # 删除保存文件目录
                    if os.path.exists(SAVED_FILES_DIR):
                        shutil.rmtree(SAVED_FILES_DIR)
                        os.makedirs(SAVED_FILES_DIR, exist_ok=True)
                    
                    # 清除元数据
                    if os.path.exists(METADATA_FILE):
                        os.remove(METADATA_FILE)
                    
                    # 清除会话状态
                    st.session_state.current_supermarket_db = None
                    st.session_state.product_df = None
                    st.session_state.supermarket_messages = []
                    
                    st.success("所有商品数据已清除")
                    st.rerun()
                except Exception as e:
                    st.error(f"清除失败: {e}")
    
    # 底部信息
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🔧 技术栈:**")
        st.markdown("• LangChain • Streamlit • FAISS • DeepSeek")
    with col2:
        st.markdown("**✨ 功能特色:**")
        st.markdown("• PDF智能问答 • 数据可视化分析")
    with col3:
        st.markdown("**🛒 超市客服:**")
        st.markdown("• 商品信息查询 • 智能推荐")
    with col4:
        st.markdown("**💾 数据管理:**")
        st.markdown("• 文件持久化 • 自动加载")

if __name__ == "__main__":
    main()