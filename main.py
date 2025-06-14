import os
from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    WebSearchTool,
    OpenAIServerModel,
)
# from langchain.agents import load_tools
from tools.arxiv_tools import (
    arxiv_paper_search, 
    arxiv_paper_filter_and_classify, 
    download_arxiv_paper
)
from tools.rag_tools import create_rag, quary_rag

load_dotenv() 

model = OpenAIServerModel(
    model_id="qwen-max",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("QWEN_KEY"),
)

agent = CodeAgent(
    tools=[arxiv_paper_search, 
           arxiv_paper_filter_and_classify, 
           download_arxiv_paper,
           create_rag,
           quary_rag],
    model=model,
    # managed_agents=[arxiv_paper_search,arxiv_paper_filter_and_classify],
    )

agent.prompt_templates["system_prompt"] = agent.prompt_templates["system_prompt"] + \
    """\n After using the quary_rag function, you need to further think and answer based on the returned result, rather than directly using the result of quary_rag as the final output.
    用户输入一般为：构建研究主题为XXX的awesome的数据库;这时你需要执行：查询某领域下，研究主题为XXX相关的论文，并保存元数据到本地的./paper_dataset/XXX/{filename}.json文件中。然后进行过滤与分类，随后根据下载论文的pdf到本地目录"./paper_dataset/XXX/pdfs"，并构建RAG数据库"./paper_dataset/XXX/chroma_langchain_db"。
   
    """

# manager_agent.run("查询AI领域下，研究主题为video generation相关的论文，并保存在本地的json文件中。然后对这个json文件进行过滤，只保留给定主题的论文")
agent.run("构建研究主题为Object detection的awesome的数据库")