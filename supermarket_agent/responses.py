from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.tools import PythonAstREPLTool

from .embeddings import init_embeddings, init_llm
from .processing import check_database_exists
from .config import DB_FILE


def get_pdf_response(user_question, db_name=None):
    if not db_name:
        return "❌ 未指定数据库"

    if not check_database_exists(db_name):
        return "❌ 请先上传PDF并处理后再查询"

    try:
        embeddings = init_embeddings()
        llm = init_llm()

        db_path = __import__("os").path.join(DB_FILE, db_name)
        new_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
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


def get_supermarket_response(user_question, db_name="supermarket_db"):
    if not check_database_exists(db_name):
        return "❌ 请先选择或上传商品CSV文件！"

    try:
        embeddings = init_embeddings()
        llm = init_llm()

        db_path = __import__("os").path.join(DB_FILE, db_name)
        new_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever(search_kwargs={"k": 5})

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个超市的智能客服助手。请根据提供的商品信息回答顾客的问题。回答时请友好并基于数据给出准确信息。如果没有相关信息请如实说明。"""),
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


def get_csv_response(query: str, session_df):
    if session_df is None:
        return "请先上传CSV文件"

    llm = init_llm()
    locals_dict = {'df': session_df}
    tools = [PythonAstREPLTool(locals=locals_dict)]

    system = f"""Given a pandas dataframe `df` answer user's query.
    Here's the output of `df.head().to_markdown()` for your reference, you have access to full dataframe as `df`:
    ```
    {session_df.head().to_markdown()}
    ```
    Give final answer as soon as you have enough data, otherwise generate code using `df` and call required tool.
    If user asks you to make a graph, save it as `plot.png`, and output GRAPH:<graph title>.
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
