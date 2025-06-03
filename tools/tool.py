import os
import json
from datetime import datetime

def generate_readme(json_path, output_path="README.md"):
    # 读取JSON文件
    print("json path",json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    # 读取json路径的倒数第二个文件夹名作为分类, ./paper_dataset\video generation\video generation_20250531.json
    
    category_name = json_path.split(os.sep)[-2] if os.sep in json_path else "Research Papers"
    # 按分类分组
    categories = {}
    for paper in papers:
        category = paper.get("classification", "Uncategorized")
        if category not in categories:
            categories[category] = []
        categories[category].append(paper)
    
    # 按分类字母顺序排序
    sorted_categories = sorted(categories.items())
    
    # 生成Markdown内容
    md_content = f"# Awesome {category_name} Research\n\n"
    md_content += "> 自动生成的论文列表，更新于: {}\n\n".format(datetime.now().strftime("%Y-%m-%d"))
    
    # 目录生成
    md_content += "## 目录\n"
    for category, _ in sorted_categories:
        md_content += f"- [{category}](#{category.lower().replace(' ', '-')})\n"
    md_content += "\n---\n\n"
    
    # 按分类添加论文
    for category, papers_in_category in sorted_categories:
        md_content += f"## {category}\n\n"
        
        # 按发布日期排序（最新优先）
        papers_in_category.sort(key=lambda x: x["published"], reverse=True)
        
        for paper in papers_in_category:
            # 格式化日期
            pub_date = datetime.strptime(paper["published"], "%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y")
            update_date = datetime.strptime(paper["updated"], "%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y")
            
            md_content += f"### [{paper['title']}]({paper['pdf_url']})\n\n"
            md_content += f"**Authors**: {', '.join(paper['authors'])}\n\n"
            md_content += f"**Published**: {pub_date} | **Updated**: {update_date} | "
            md_content += f"**Arxiv ID**: [{paper['arxiv_id']}]({paper['pdf_url']})\n\n"
            md_content += "**Abstract**:\n> " + paper['abstract'].replace('\n', '\n> ') + "\n\n"
            md_content += "---\n\n"
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"README generated successfully at: {output_path}")

# 使用示例
if __name__ == "__main__":
    generate_readme("image_generation_20250531.json")