from smolagents import tool
import arxiv
import json
import os
import glob
from datetime import datetime
from typing import List, Dict
from tools.tool import generate_readme
import requests
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv() 


BATCH_SIZE = 20  # 每次处理BATCH_SIZE篇论文
client = AsyncOpenAI(
    api_key=os.getenv("QWEN_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

@tool
def arxiv_paper_search(topic: str, keywords: str) -> str:
    """
    Searches arXiv for academic papers matching a specified topic and keywords in the title.
    Returns a status message indicating where the data was saved or if it already exists.
    
    Args:
        topic (str): The research topic or category to search in arXiv (e.g., 'Computer Science', 'Physics').
        keywords (str): A string of keywords to search in the paper titles, separated by commas (e.g., 'quantum, computing').
    
    Returns:
        str: paper_metadata_saved_path or "Data already exists"
    """
    result = fetch_arxiv_papers(topic, keywords)
    return result

def fetch_arxiv_papers(topic: str, keywords: str, max_results: int = 10) -> str:
    """
    从arXiv API获取指定主题和关键词的论文数据，并保存到JSON文件
    
    参数:
    topic (str): 研究主题 (如："quantum computing")
    keywords (str): 关键词列表 (如："quantum")
    max_results (int): 最大返回结果数量
    
    返回:
    str: 操作状态信息
    """
    # 创建保存目录
    base_dir = ".\\paper_dataset"
    keyword_dir = os.path.join(base_dir, keywords)
    os.makedirs(keyword_dir, exist_ok=True)
    
    # 获取当前日期
    current_date = datetime.now().strftime("%Y%m%d")
    filename = f"{keywords}_{current_date}.json"
    file_path = os.path.join(keyword_dir, filename)
    
    # 检查文件是否已存在
    if os.path.exists(file_path):
        return f"论文数据已存在: {file_path}"
    
    # 构造查询字符串
    # query = f"cat:{topic} AND ti:{keywords}"
    query = f"{keywords}"

    new_papers = []
    try:
        search = arxiv.Search(
            query = query,
            max_results = max_results,
            sort_by = arxiv.SortCriterion.Relevance
            )
        # 提取论文信息
        for result in search.results():
            print(f"Found paper: {result.title}")
            paper_info = {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "published": result.published.strftime("%Y-%m-%d %H:%M:%S"),
                "updated": result.updated.strftime("%Y-%m-%d %H:%M:%S"),
                "arxiv_id": result.entry_id.split('/')[-1],
                "pdf_url": result.pdf_url,
                "filtered": False,  # 默认未过滤
                "summary":"",
                "classification": "",
            }
            new_papers.append(paper_info)
            
    except Exception as e:
        return f"获取数据时出错: {e}"
    
    # 查找该关键词目录下的所有JSON文件
    existing_files = glob.glob(os.path.join(keyword_dir, "*.json"))
    all_papers = []
    print(f"合并现有{len(existing_files)}个文件的数据...")
    # 读取并合并所有现有文件
    for file in existing_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_papers.extend(data)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    # 去重处理 (基于arxiv_id)
    existing_ids = {paper['arxiv_id'] for paper in all_papers}
    unique_new_papers = [paper for paper in new_papers if paper['arxiv_id'] not in existing_ids]
    
    # 合并新旧论文
    all_papers.extend(unique_new_papers)
    
    # 保存到新文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_papers, f, ensure_ascii=False, indent=4)
    except Exception as e:
        return f"保存JSON文件时出错: {e}"
    
    # 删除旧文件 (保留最新文件)
    for file in existing_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"删除旧文件 {file} 时出错: {e}")
    
    return f"{file_path}"

@tool
def download_arxiv_paper(paper_metadata_saved_path:str) -> str:
    """
    Downloads the all PDF of a papers from meta data paper json file. save_dir = os.path.join(os.path.dirname(paper_metadata_saved_path), "pdfs").
    
    Args:
        paper_metadata_saved_path (str): The path containing the papers information. The paper_metadata_saved_path can get it from return of arxiv_paper_search function.
        such as :
            paper_metadata_saved_path = arxiv_paper_search(topic="Computer Science", keywords="Object detection")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            Step 2: Filter and classify the downloaded papers.                                                                                                                                                                          
            filter_result = arxiv_paper_filter_and_classify(keywords="Object detection", research_content="Object detection", topic="")                                                                                                   
            download_papers = download_arxiv_paper(paper_metadata_saved_path=paper_metadata_saved_path)
    
    Returns:
        str: A message indicating whether the download was successful or not.
    """
    with open(paper_metadata_saved_path, 'r') as json_file:
        loaded_data = json.load(json_file)
    save_dir = os.path.join(os.path.dirname(paper_metadata_saved_path), "pdfs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for paper in loaded_data:
        paper_id = paper.get("arxiv_id")
        pdf_url = paper.get("pdf_url")
        paper_save_path = os.path.join(save_dir, f"{paper_id}.pdf")
        print(download_file(pdf_url, paper_save_path))
    return f"已下载论文PDF到: {save_dir}"
        

def download_file(url: str, save_path: str) -> str:
    """
    Downloads a file from a given URL and saves it to the specified path.
    
    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path where the file should be saved.
    
    Returns:
        str: A message indicating whether the download was successful or not.
    """
    if os.path.exists(save_path):
        return f"文件已存在: {save_path}"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return f"文件已下载并保存到: {save_path}"
    except Exception as e:
        return f"下载文件时出错: {str(e)}"


@tool
def arxiv_paper_filter_and_classify(keywords: str, research_content: str, topic: str="" ) -> str:
    """
    Filters papers in saved JSON files using an LLM to determine relevance to the specified topic and keywords.
    
    This function:
    1. Locates the latest JSON file for the given keywords in the paper_dataset directory
    2. Identifies papers that haven't been filtered yet (filtered ≠ True)
    3. Uses an LLM to assess each paper's relevance based on title and abstract
    4. For relevant papers: sets 'filtered' = True
    5. For irrelevant papers: deletes the entire paper entry
    6. Skips papers that have already been filtered
    
    Args:
        keywords (str): The name of the parent directory of the json file
        research_content (str): A string whose value can be the same as keywords or a further detailed description of keywords.
        topic (str): defaults to empty string
    
    Returns:
        str: Operation status message indicating:
             - Path to updated JSON file
             - Number of relevant papers kept
             - Number of irrelevant papers deleted
             - "All papers already filtered" if no new papers to process
             - Error message if issues occur
    """
    # 创建目录路径
    base_dir = "./paper_dataset"
    keyword_dir = os.path.join(base_dir, keywords)
    
    # 检查目录是否存在
    if not os.path.exists(keyword_dir):
        return f"未找到关键词目录: {keyword_dir}"
    
    # 查找最新的JSON文件
    json_files = glob.glob(os.path.join(keyword_dir, f"{keywords}_*.json"))
    if not json_files:
        return f"未找到关键词 '{keywords}' 的论文文件"
    
    # 按修改时间排序获取最新文件
    latest_file = max(json_files, key=os.path.getmtime)
    
    try:
        # 读取现有论文数据
        with open(latest_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        # 获取需要过滤的论文（未过滤的）
        papers_to_filter = [
            paper for paper in papers 
            if not paper.get("filtered", False)
        ]
        
        if not papers_to_filter:
            return f"所有论文已过滤: {latest_file}"
        
        # 使用大模型过滤论文并获取相关性结果
        relevance_results = asyncio.run(
            filter_papers_with_llm(topic, research_content, papers_to_filter)
        )
        
        # 创建新论文列表（只保留相关论文）
        new_papers = []
        deleted_count = 0
        kept_count = 0
        
        # 处理所有论文（包括已过滤和未过滤的）
        for paper in papers:
            if not paper.get("filtered", False):
                arxiv_id = paper["arxiv_id"]
                result = relevance_results.get(arxiv_id)
                
                if result and result["relevant"]:
                    paper["filtered"] = True
                    paper["classification"] = result["classification"]
                    paper["summary"] = result["summary"]
                    new_papers.append(paper)
                    kept_count += 1
                else:
                    deleted_count += 1
        
        # 保存更新后的数据（已删除不相关论文）
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(new_papers, f, ensure_ascii=False, indent=4)
            
        generate_readme(latest_file, os.path.join(keyword_dir, "README.md"))
        return (
            f"论文过滤完成: {latest_file}\n"
            f"保留相关论文: {kept_count} 篇\n"
            f"删除不相关论文: {deleted_count} 篇"
        )
    
    except Exception as e:
        return f"处理论文过滤时出错: {str(e)}"

async def filter_papers_with_llm(topic: str, keywords: str, papers: List[Dict]) -> Dict[str, Dict]:
    """
    使用大模型批量过滤论文相关性
    
    参数:
    topic (str): 研究主题
    keywords (str): 关键词
    papers (List[Dict]): 待过滤的论文列表
    
    返回:
    Dict[str, Dict]: 论文ID到相关性和分类结果的映射
        {arxiv_id: {"relevant": bool, "classification": str}}
    """
    # 分批处理论文
    relevance_map = {}
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i:i+BATCH_SIZE]
        try:
            batch_result = await assess_batch_relevance(topic, keywords, batch)
            relevance_map.update(batch_result)
        except Exception as e:
            print(f"批量处理论文时出错: {str(e)}")
            # 出错时标记整批为不相关
            for paper in batch:
                arxiv_id = paper["arxiv_id"]
                relevance_map[arxiv_id] = {"relevant": False, "classification": "Error"}
    
    return relevance_map

async def assess_batch_relevance(topic: str, keywords: str, batch: List[Dict]) -> Dict[str, Dict]:
    """
    使用大模型批量评估论文相关性
    
    参数:
    topic (str): 研究主题
    keywords (str): 关键词
    batch (List[Dict]): 论文批次
    
    返回:
    Dict[str, Dict]: 论文ID到结果的映射
    """
    # 构建系统提示
    system_prompt = (
        "你是一个学术研究助手，负责批量评估论文相关性。\n"
        f"研究主题: {topic}\n"
        f"关键词: {keywords}\n\n"
        "请根据每篇论文的标题和摘要，执行以下操作：\n"
        "1. 判断论文是否与研究主题或关键词相关\n"
        "2. 对相关论文进行大致分类，不要过细导致没有相同类别的文章，类别数大概为2-3个（如：'Multimodal Instruction Tuning','Multimodal Hallucination'）\n"
        "3. 不相关论文分类标记为'Unrelated'\n\n"
        "4. 根据abstract和title，编写一段简短的summary\n"
        "返回格式要求：\n"
        "使用严格JSON格式，键为arxiv_id，值为包含两个字段的对象：\n"
        "- 'relevant': 布尔值（true/false）\n"
        "- 'classification': 字符串\n"
        "- 'summary': 字符串\n"
        "示例：\n"
        "{\"1234.56789\": {\"relevant\": true, \"classification\": \"Vision-Language Models\",  \"summary\": \"XXX XXX xXX\"}, "
        "\"2345.67890\": {\"relevant\": false, \"classification\": \"Unrelated\", \"summary\": \"XXX XXX xXX\"}}"
    )

    # 构建用户提示
    user_prompt = "请评估以下论文：\n"
    for paper in batch:
        user_prompt += (
            f"\narxiv_id: {paper['arxiv_id']}\n"
            f"标题: {paper['title']}\n"
            f"摘要: {paper['abstract']}\n"
            "----------------------------------------"
        )

    try:
        # 调用大模型
        response = await client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
            temperature=0.1,  # 降低随机性
            response_format={"type": "json_object"}  # 要求返回JSON格式
        )

        # 解析响应
        result_json = response.choices[0].message.content.strip()
        print(f"模型返回结果: {result_json}")
        
        # 解析JSON结果
        results = json.loads(result_json)
        
        # 验证结果格式
        validated_results = {}
        for arxiv_id, data in results.items():
            if isinstance(data, dict) and "relevant" in data and "classification" in data:
                validated_results[arxiv_id] = {
                    "relevant": bool(data["relevant"]),
                    "classification": data["classification"],
                    "summary": data["summary"]
                }
            else:
                validated_results[arxiv_id] = {"relevant": False, "classification": "Invalid Format"}
        
        return validated_results

    except json.JSONDecodeError:
        print("JSON解析失败，返回内容:", result_json)
        # 返回空结果（整批视为不相关）
        return {paper["arxiv_id"]: {"relevant": False, "classification": "JSON Error"} for paper in batch}
        
    except Exception as e:
        print(f"批量评估出错: {str(e)}")
        # 返回空结果（整批视为不相关）
        return {paper["arxiv_id"]: {"relevant": False, "classification": "Error"} for paper in batch}


if __name__ == "__main__":
    # download_arxiv_paper("D:\\code\\myagent\\paper_dataset\\video generation\\video generation_20250602.json")
    # print(create_rag("./paper_dataset/test/pdfs", "./paper_dataset/test/chroma_langchain_db"))
    # print(quary_rag("用的语言模型有哪些？", "./rag_db"))
    print(arxiv_paper_search("AI","video generation"))
