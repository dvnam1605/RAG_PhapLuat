from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

embeddings = GPT4AllEmbeddings(model_file="model/all-MiniLM-L6-v2-f16.gguf")

def process_text_files(text_path: str):

    loader = DirectoryLoader(
        text_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=1024) # Xem xét lại chunk_size cho phù hợp với model và dữ liệu
    chunks = splitter.split_documents(documents)
    return chunks

chunk = process_text_files("output_texts")
vector_store = FAISS.from_documents(
    chunk,
    embeddings
)
path_db = "vector_store/faiss"
vector_store.save_local(path_db)
