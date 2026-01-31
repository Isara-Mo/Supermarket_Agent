import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from data_manager import DB_FILE
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer
import contextvars

# 定义一个上下文变量，默认值为 0
complexity_context = contextvars.ContextVar("complexity", default=0)
# ======================
# 模块级“私有”变量
# ======================
_tokenizer = None
_session = None
_qwen_fast_model = ChatTongyi(model="qwen-flash",api_key="")                    #api_key调用暂时还不太规范
_qwen_max_model = ChatTongyi(model="qwen3-max",api_key='')
def init_embeddings(dashscope_api_key):
    return DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=dashscope_api_key)

##文本分类模块
def init_model(
    model_name="bert-base-chinese",
    onnx_path="bert_classifier.onnx",
    providers=None
):
    """
    在程序启动时调用一次
    """
    global _tokenizer, _session

    if providers is None:
        providers = ["CPUExecutionProvider"]

    _tokenizer = BertTokenizer.from_pretrained(model_name)
    _session = ort.InferenceSession(onnx_path, providers=providers)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(text):
    if _tokenizer is None or _session is None:
        raise RuntimeError("Model not initialized, call init_model() first")
    inputs = _tokenizer(
        text,
        return_tensors="np",      # 生成 numpy
        padding="max_length",
        truncation=True,
        max_length=256
    )

    # 🔴 关键：显式转成 int64
    input_ids = inputs["input_ids"].astype("int64")
    attention_mask = inputs["attention_mask"].astype("int64")

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    logits = _session.run(None, ort_inputs)[0]
    probs = softmax(logits[0])
    pred = int(np.argmax(probs))

    return pred, probs

##
@wrap_model_call
def dynamic_deepseek_routing(request: ModelRequest, handler) -> ModelResponse:
    """
    根据对话复杂度动态选择 DeepSeek 模型：
    - 复杂：deepseek-reasoner
    - 简单：deepseek-chat
    """
    # 根据预测结果选择模型
    current_pred = complexity_context.get()
    if current_pred == 1:  # 如果是复杂问题
        request.model = _qwen_max_model
    else:  # 如果是简单问题
        request.model = _qwen_fast_model
    
    print(f"此次的模型: {request.model}")
    # 调用被包裹的下游（真正的模型调用）
    return handler(request)
def _get_last_user_text(messages) -> str:
        """从消息列表中取最近一条用户消息文本（无则返回空串）"""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                # content 可能是纯字符串或富内容；这里只处理为字符串的常见情况
                return m.content if isinstance(m.content, str) else ""
        return ""
def _get_last_text(messages) -> str:
    """如果 messages 的最后一条是 HumanMessage，则返回1，否则0"""
    if not messages:
        return 0

    last = messages[-1]
    if isinstance(last, HumanMessage):
        return 1 if isinstance(last.content, str) else 0

    return 0

def create_supermarket_agent(dashscope_api_key, db_name):
    """【工厂函数】负责从零创建一个 Agent 对象"""
    # 1. 初始化模型
    llm = ChatTongyi(model="qwen-plus")
    embeddings = init_embeddings(dashscope_api_key)
    
    # 2. 加载向量数据库
    db_path = os.path.join(DB_FILE, db_name)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    # 3. 创建工具
    tool = create_retriever_tool(
        db.as_retriever(search_kwargs={"k": 5}), 
        "product_search", 
        "搜索商品信息，包括名称、价格、库存、描述等"
    )
    
    # 4. 构建 Agent
    agent = create_agent(
        model=llm, 
        tools=[tool], 
        system_prompt="你是一个超市客服助手。根据商品数据给出准确友好的回答。如果找不到商品，请礼貌地说明。",
        middleware=[dynamic_deepseek_routing]
    )
    return agent

def run_supermarket_agent(agent, user_question):
    """【执行函数】直接利用传入的 agent 对象进行推理"""
    try:
        # 使用 BERT 模型预测复杂度
        global pred, probs
        pred, probs = predict(user_question)
        complexity_context.set(pred)
        print(f"[BERT推理] 预测标签: {pred}, 预测概率: {probs}")

        response = agent.invoke({"messages": [("user", user_question)]})
        return response["messages"][-1].content
    except Exception as e:
        return f"❌ 对话执行出错: {str(e)}"