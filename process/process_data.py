import os
import re
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- CẤU HÌNH ---
DATA_PATH = "crawler/output_texts"
DB_PATH = "vector_store/faiss"
EMBEDDING_MODEL_PATH = "model/all-MiniLM-L6-v2-f16.gguf"
MAX_CHUNK_SIZE = 1200  # Kích thước tối đa cho một chunk (tính bằng ký tự)
CHUNK_OVERLAP = 300    # Độ chồng lấn giữa các chunk bị cắt

def preprocess_text(text: str) -> str:
    """
    Hàm tiền xử lý văn bản:
    - Nối các dòng bị ngắt không tự nhiên.
    - Chuẩn hóa khoảng trắng.
    - Loại bỏ các dòng trống thừa.
    """
    
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def build_vector_store():
    
    print(f"Đang tải các file từ thư mục: '{DATA_PATH}'...")
    if not os.path.exists(DATA_PATH):
        print(f"Lỗi: Thư mục '{DATA_PATH}' không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
        return
        
    loader = DirectoryLoader(
        DATA_PATH, glob="*.txt", loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}, show_progress=True,
    )
    documents = loader.load()
    if not documents:
        print(f"Lỗi: Không tìm thấy file .txt nào trong '{DATA_PATH}'.")
        return
    print(f"Đã tải thành công {len(documents)} tài liệu.")

    print("\nBắt đầu quá trình tiền xử lý văn bản...")
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)
    print("Tiền xử lý hoàn tất.")

    print("\nBắt đầu quá trình chia chunk theo ký tự...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "], 
        length_function=len
    )
    all_chunks = text_splitter.split_documents(documents)
    
    if not all_chunks:
        print("Lỗi: Không tạo được chunk nào từ các tài liệu.")
        return
        
    print(f"Hoàn tất! Đã tạo tổng cộng {len(all_chunks)} chunks.")
    
    print("\nĐang khởi tạo mô hình embeddings...")

    model_path= "models/vietnamese-bi-encoder"
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    
    # embeddings = GPT4AllEmbeddings(
    #     model_file=EMBEDDING_MODEL_PATH,
    # )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
)
    
    print("Đang tạo vector store từ các chunks (quá trình này có thể mất vài phút)...")
    vector_store = FAISS.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )
    
    if not os.path.exists("vector_store"):
        os.makedirs("vector_store")
    vector_store.save_local(DB_PATH)
    print(f"\n🎉 Vector store đã được tạo và lưu thành công tại: '{DB_PATH}'")

if __name__ == "__main__":
    build_vector_store()