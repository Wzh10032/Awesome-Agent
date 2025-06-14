# Awesome-Agent

Awesome-agent 是一个基于[Smolagents](Smolagents)实现的智能代理项目，可以根据关键词从Arxiv上爬取相应论文，并进行分类整理形成类似Awesome-XXX的README文件（例如[Object detection](./paper_dataset/Object%20detection/)），此外还实现了简单的RAG功能，用于回答相应论文相关的问题。



## 安装

```bash
git clone https://github.com/Wzh10032/Awesome-agent.git
cd Awesome-agent
pip install -r requirements.txt
```

## 使用方法

配置[HF_TOKEN](https://huggingface.co/docs/hub/security-tokens)和[QWEN_KEY](https://bailian.console.aliyun.com/?utm_content=se_1021228191&gclid=CjwKCAjwl_XBBhAUEiwAWK2hzpBANEM5LnKgFftyttdhpOJ2wBdsGClZBKJmrIIuoe6bowGE5qoubBoCcKYQAvD_BwE&tab=model#/api-key)。

运行主程序：

```bash
python main.py
```
运行gradio_gui程序
```bash
python gradio_gui_show.py
```
修改fetch_arxiv_papers函数中的max_results参数，可以修改爬取的论文数量。

## 运行结果
<img src=".\assets\result.png" width="800">

## Acknowledgement
我们采用了以下开源项目:
- [Smolagents](https://github.com/smol-ai/smolagents)
- [zero_nlp](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/smolagent_tutorial)
- [arxiv.py](https://github.com/lukasschwab/arxiv.py)
