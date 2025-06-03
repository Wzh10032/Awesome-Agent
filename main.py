import os
from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    WebSearchTool,
    OpenAIServerModel,
)
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
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

agent = ToolCallingAgent(
    tools=[arxiv_paper_search, 
           arxiv_paper_filter_and_classify, 
           download_arxiv_paper,
           create_rag,
           quary_rag],
    model=model,
    # managed_agents=[arxiv_paper_search,arxiv_paper_filter_and_classify],
    )

agent.prompt_templates["system_prompt"] = agent.prompt_templates["system_prompt"] + "\nAfter using the quary_rag function, you need to further think and answer based on the returned result, rather than directly using the result of quary_rag as the final output."

# manager_agent.run("查询AI领域下，研究主题为video generation相关的论文，并保存在本地的json文件中。然后对这个json文件进行过滤，只保留给定主题的论文")
agent.run("查询AI领域下，研究主题为Large Lanuage Model相关的论文，并保存在本地的json文件中。然后进行过滤与分类，随后根据下载论文的pdf到本地，最后将这些论文构建RAG数据库，并检索回答问题“常用的语言模型有哪些？")