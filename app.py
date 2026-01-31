import streamlit as st
import pandas as pd
import os
import shutil
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')

# 导入自定义模块
import data_manager as dm
import ai_agent as ai

load_dotenv(override=True)

# 基础配置
dashscope_api_key = os.getenv("dashscope_api_key")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dm.init_dirs()

st.set_page_config(page_title="智能超市个性化客服", page_icon="🤖", layout="wide")

# --- CSS 样式 (保持不变) ---
st.markdown("""
<style>
    :root { --primary-color: #1f77b4; --secondary-color: #ff7f0e; --success-color: #2ca02c; --supermarket-color: #28a745; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .main-header { background: linear-gradient(90deg, #1f77b4, #ff7f0e, #28a745); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: bold; text-align: center; margin-bottom: 2rem; }
    .info-card { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 4px solid var(--primary-color); }
    .success-card { background: linear-gradient(135deg, #e8f5e8, #f0f8f0); border-left: 4px solid var(--success-color); }
    .warning-card { background: linear-gradient(135deg, #fff8e1, #fffbf0); border-left: 4px solid var(--warning-color); }
    .supermarket-card { background: linear-gradient(135deg, #e8f8f0, #f0fff4); border-left: 4px solid var(--supermarket-color); }
    .stButton > button { background: linear-gradient(45deg, #1f77b4, #2196F3); color: white; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'supermarket_messages' not in st.session_state: st.session_state.supermarket_messages = []
    if 'product_df' not in st.session_state: st.session_state.product_df = None
    if 'current_supermarket_db' not in st.session_state: st.session_state.current_supermarket_db = None
    # 新增：用于缓存整个 Agent 对象
    if 'supermarket_agent' not in st.session_state: st.session_state.supermarket_agent = None
    # 新增：记录当前 Agent 绑定的数据库 ID，用于判断是否需要重建
    if 'agent_db_id' not in st.session_state: st.session_state.agent_db_id = None

def main():
    init_session_state()
    ai.init_model(
    model_name="bert-base-chinese",
    onnx_path="bert_classifier_RAG.onnx",
    #providers=["TensorrtExecutionProvider","CUDAExecutionProvider", "CPUExecutionProvider"]
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
    st.markdown('<h1 class="main-header">🤖 智能超市个性化客服</h1>', unsafe_allow_html=True)
    
    saved_dbs = dm.check_saved_databases()
    tab1, tab2 = st.tabs(["🛒 超市智能客服", "📁 数据管理"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🛒 超市智能客服")
            current_db = st.session_state.current_supermarket_db

            # --- 核心修改：检查并创建 Agent ---
            if current_db and dm.check_database_exists(current_db):
                # 如果 Agent 还没创建，或者当前选择的数据库变了
                if st.session_state.supermarket_agent is None or st.session_state.agent_db_id != current_db:
                    with st.status("🛠️ 正在初始化智能助手...", expanded=False) as status:
                        st.write("加载模型与索引...")
                        st.session_state.supermarket_agent = ai.create_supermarket_agent(dashscope_api_key, current_db)
                        st.session_state.agent_db_id = current_db
                        status.update(label="✅ 助手就绪！", state="complete", expanded=False)
                
                st.markdown(f'<div class="info-card supermarket-card">🛒 当前使用数据库: {current_db}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card warning-card">⚠️ 请先在右侧上传或选择数据库</div>', unsafe_allow_html=True)

            # 欢迎消息
            if not st.session_state.supermarket_messages and current_db:
                welcome_msg = "🛒 欢迎来到智能超市！我是您的专属客服助手，可以帮您查找商品信息、价格等。"
                st.session_state.supermarket_messages.append({"role": "assistant", "content": welcome_msg})

            # 渲染历史对话
            for msg in st.session_state.supermarket_messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

            # 聊天输入处理
            can_chat = current_db and st.session_state.supermarket_agent is not None
            if query := st.chat_input("🛒 询问商品信息...", disabled=not can_chat):
                st.session_state.supermarket_messages.append({"role": "user", "content": query})
                with st.chat_message("user"): st.markdown(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("🔍 正在查找..."):
                        # 直接调用持久化的 Agent
                        res = ai.run_supermarket_agent(st.session_state.supermarket_agent, query)
                        st.markdown(res)
                        st.session_state.supermarket_messages.append({"role": "assistant", "content": res})

        with col2:
            st.markdown("### 🏪 商品数据管理")
            if saved_dbs:
                for i, db_info in enumerate(saved_dbs):
                    is_current = st.session_state.current_supermarket_db == db_info["db_name"]
                    cola, colb = st.columns([3, 1])
                    cola.write(f"{'🟢' if is_current else '⚪'} **{db_info['original_name']}**")
                    if colb.button("选择", key=f"sel_{i}", disabled=is_current):
                        st.session_state.current_supermarket_db = db_info["db_name"]
                        st.session_state.product_df = dm.load_saved_csv(db_info["filename"])
                        st.session_state.supermarket_messages = []
                        # 切换数据库时清空旧 Agent，强制下次运行时重建
                        st.session_state.supermarket_agent = None 
                        st.rerun()

            product_csv = st.file_uploader("📤 上传商品CSV", type='csv', key="prod_csv")
            if product_csv:
                st.session_state.product_df = pd.read_csv(product_csv)
                st.success("✅ 数据加载成功!")
                if st.button("🚀 保存并处理", use_container_width=True):
                    with st.spinner("💾 处理中..."):
                        fname, fpath = dm.save_csv_file(product_csv, "product")
                        db_n = dm.load_metadata()[fname]["db_name"]
                        # 处理时临时初始化 embeddings
                        success, res = dm.process_product_csv(st.session_state.product_df, db_n, ai.init_embeddings(dashscope_api_key))
                        if success:
                            st.session_state.current_supermarket_db = db_n
                            st.session_state.supermarket_messages = []
                            st.session_state.supermarket_agent = None # 清理 Agent 触发重建
                            st.success(f"✅ 已创建 {res} 个数据块")
                            st.rerun()

    with tab2:
        st.markdown("### 📁 数据管理中心")
        if saved_dbs:
            for i, db_info in enumerate(saved_dbs):
                with st.expander(f"📄 {db_info['original_name']}", expanded=False):
                    col_info, col_ops = st.columns([2, 1])
                    with col_info:
                        st.caption(f"• 类型: {db_info['file_type']} | 上传: {db_info['upload_time']}")
                    with col_ops:
                        if st.button("🗑️ 删除文件", key=f"del_tab2_{i}", type="secondary", use_container_width=True):
                            shutil.rmtree(os.path.join(dm.DB_FILE, db_info['db_name']), ignore_errors=True)
                            if os.path.exists(db_info['file_path']): os.remove(db_info['file_path'])
                            meta = dm.load_metadata()
                            if db_info['filename'] in meta:
                                del meta[db_info['filename']]
                                dm.save_metadata(meta)
                            # 如果删的是当前用的数据库，重置状态
                            if st.session_state.current_supermarket_db == db_info['db_name']:
                                st.session_state.current_supermarket_db = None
                                st.session_state.supermarket_agent = None
                            st.rerun()
                        if st.button("👀 预览数据", key=f"preview_{i}"):
                            try:
                                df = dm.load_saved_csv(db_info['filename'])
                                if df is not None:
                                    # 显示数据的前几行
                                    st.dataframe(df.head())
                                    st.write(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
                                else:
                                    st.error("无法加载数据")
                            except Exception as e:
                                st.error(f"预览失败: {str(e)}")
        else:
            st.info("🗂️ 暂无保存的数据文件")

if __name__ == "__main__":
    main()