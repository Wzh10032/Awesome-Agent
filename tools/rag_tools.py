import os
from smolagents import tool
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_embedding():
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
def get_vector_store(embeddings=None, persist_directory="./chroma_langchain_db"):    
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
    )
    return vector_store

async def load_pdf(file_path):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

@tool
def create_rag(paper_dir:str) -> str:
    """
    Creates a RAG (Retrieval-Augmented Generation) knowledge base by processing academic papers 
    and storing document embeddings in a vector database.
    
    This function processes all PDF files in the specified directory, splits them into manageable 
    text chunks, embeds them using a pretrained model, and stores the results in a vector database. 
    If the target persist directory already exists, the operation is skipped.
    The RAG database is stored in os.path.join(os.path.dirname(paper_dir), "chroma_langchain_db")

    Args:
        paper_dir (str): Directory path containing PDF files to process. such as "./paper_dataset/Object detection/pdfs"

    Returns:
        str: Success message indicating creation status, including directory location.
    """
    if "pdfs" not in paper_dir:
        persist_directory = os.path.join(paper_dir, "chroma_langchain_db")
    else:
        persist_directory = os.path.join(os.path.dirname(paper_dir), "chroma_langchain_db")
    if os.path.exists(persist_directory):
        return f"Vector store already exists at {persist_directory}. No need to create again."
    
   
    embeddings = get_embedding()
    vector_store = get_vector_store(embeddings=embeddings, persist_directory=persist_directory)
   
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 列举paper_dir目录下所有的PDF文件
    pages = []
    for f in os.listdir(paper_dir):
        if f.endswith('.pdf'):
            pdf_path = os.path.join(paper_dir, f)
            pages = asyncio.run(load_pdf(pdf_path))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(pages)
    _ = vector_store.add_documents(documents=all_splits)
    return f"Vector store created at {persist_directory} with {len(all_splits)} document chunks."

@tool
def quary_rag(query:str, persist_directory:str) -> str:
    """
    Searches the RAG knowledge base for documents relevant to the given query.
    Performs similarity search against the vector database using the query text, 
    returning the top 5 most relevant document excerpts.
    This function is only used to provide relevant context. Answering the question still requires you to reorganize your answer.

    Args:
        query (str): Text query to search against the knowledge base.
        persist_directory (str): Directory path where the vector database is stored.

    Returns:
        str: Formatted string containing up to 5 relevant document excerpts, with headers 
            indicating their rank in search results.
    """
    embeddings = get_embedding()
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector store directory does not exist: {persist_directory}")
    vector_store = get_vector_store(embeddings=embeddings, persist_directory=persist_directory)
    
    docs = vector_store.similarity_search(query, k=5)
    return "\nRetrieved documents:\n" + "".join(
        [f"\n\n===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
    )

if  __name__ == "__main__":
    print(create_rag(".//paper_dataset//image Super-Resolution/pdfs", ".//paper_dataset//image Super-Resolution//chroma_langchain_db"))