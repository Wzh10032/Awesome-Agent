import os
import json
import re
from datetime import datetime
from urllib.parse import urlparse

def extract_code_link(abstract):
    """从abstract中提取并清理开源代码链接"""
    # 匹配所有可能的URL模式（包括中间有空格的）
    url_patterns = [
        r'https?:\s*//\S+',  # 处理协议部分有空格的情况
        r'http\s*://\S+',     # 处理http后跟空格的情况
        r'www\.\S+\.\S+',     # 匹配www开头的网址
        r'\S+\.(com|org|net|io|ai|github|gitlab)\S*'  # 匹配常见域名
    ]
    
    # 尝试所有模式，直到找到匹配项
    for pattern in url_patterns:
        matches = re.findall(pattern, abstract)
        if matches:
            # 选择最长的匹配项（通常是最完整的链接）
            raw_url = max(matches, key=len)
            
            # 基本清理：移除所有空格和常见结尾标点
            clean_url = raw_url.replace(' ', '')
            clean_url = re.sub(r'[.,;:!?)\'"]+$', '', clean_url)
            
            # 验证是否为有效URL
            if is_valid_url(clean_url):
                return clean_url
    
    return None

def is_valid_url(url):
    """检查是否为有效URL"""
    try:
        result = urlparse(url)
        # 基本验证：需要协议和网络位置
        return all([result.scheme, result.netloc])
    except:
        return False

def generate_readme(json_path, output_path="README.md"):
    # 读取JSON文件
    print(f"JSON路径: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # 获取分类名称
    category_name = json_path.split(os.sep)[-2] if os.sep in json_path else "Research Papers"
    
    # 按分类分组
    categories = {}
    for paper in papers:
        category = paper.get("classification", "Uncategorized")
        categories.setdefault(category, []).append(paper)
    
    # 生成Markdown内容
    md_content = f"# 📚 Awesome {category_name} Research\n\n"
    md_content += f"> 🗓️ 自动生成的论文列表 | 最后更新: {datetime.now().strftime('%Y-%m-%d')}\n\n"
    
    # 目录生成
    md_content += "## 🔍 目录\n"
    for category in sorted(categories.keys()):
        md_content += f"- [{category}](#{category.lower().replace(' ', '-')})\n"
    md_content += "\n---\n\n"
    
    # 按分类添加论文
    for category, papers_in_category in sorted(categories.items()):
        md_content += f"## 📁 {category}\n\n"
        
        # 按发布日期排序（最新优先）
        papers_in_category.sort(key=lambda x: x["published"], reverse=True)
        
        for paper in papers_in_category:
            # 格式化日期
            pub_date = datetime.strptime(paper["published"], "%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y")
            update_date = datetime.strptime(paper["updated"], "%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y")
            
            md_content += f"### 📄 [{paper['title']}]({paper['pdf_url']})\n\n"
            md_content += f"👤 **Authors**: {', '.join(paper['authors'])}\n\n"
            md_content += f"📅 **Published**: `{pub_date}` | 🔄 **Updated**: `{update_date}` | "
            md_content += f"🆔 **Arxiv ID**: [{paper['arxiv_id']}]({paper['pdf_url']})\n\n"
            
            # 提取开源代码链接
            code_link = extract_code_link(paper['abstract'])
            if code_link:
                # 确保URL有协议前缀
                if not code_link.startswith('http'):
                    code_link = 'https://' + code_link.lstrip('/')
                md_content += f"💻 **Code**: [Link]({code_link})\n\n"
            
            md_content += "🤖 **AI Summary**:\n> " + paper['summary'].replace('\n', '\n> ') + "\n\n"
            md_content += "---\n\n"
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"README 生成成功: {output_path}")
# 使用示例
if __name__ == "__main__":
    generate_readme("D:\code\Awesome-agent\paper_dataset\Object detection\Object detection_20250614.json", output_path="Object detection.md")