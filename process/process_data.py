import os
import re
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "crawler\output_texts"
DB_PATH = "vector_store/faiss"
EMBEDDING_MODEL_PATH = "model/all-MiniLM-L6-v2-f16.gguf"


def refined_clean_legal_text(raw_text: str) -> str:
    
    lines = raw_text.split('\n')
    processed_lines = []
    temp_line = ""
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        
        temp_line = (temp_line + " " + stripped_line).strip()
        
        # Coi một dòng là hoàn chỉnh nếu nó kết thúc bằng dấu câu mạnh hoặc là tiêu đề
        if temp_line.endswith(('.', ':', ';', '”', '/.')) or temp_line.isupper():
            processed_lines.append(temp_line)
            temp_line = ""
    if temp_line:
        processed_lines.append(temp_line)

    text = "\n".join(processed_lines)

    patterns_to_remove = [
        # Xóa Tiêu ngữ và Quốc hiệu
        r'CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\s*Độc lập - Tự do - Hạnh phúc',
        # Xóa các dòng gạch ngang trang trí
        r'^\s*-{5,}\s*$',
        # Xóa khối Chữ ký ở cuối văn bản
        r'TM\.\s*ỦY BAN NHÂN DÂN\s*CHỦ TỊCH\s*[\w\s]+',
        # Xóa khối "Nơi nhận" ở cuối văn bản
        r'Nơi nhận:[\s\S]*?(?=\Z)', # \Z khớp với cuối cùng của chuỗi
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
    # 3. Chuẩn hóa khoảng trắng và các dòng trống thừa
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text.strip()


def build_vector_store():
  
    print(f"Đang tải các file từ thư mục: '{DATA_PATH}'...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    if not documents:
        print(f"Lỗi: Không tìm thấy file .txt nào trong thư mục '{DATA_PATH}'. Vui lòng kiểm tra lại.")
        return
    print(f"Đã tải {len(documents)} tài liệu.")

    print("Đang làm sạch nội dung các tài liệu...")
    for doc in documents:
        
        doc.page_content = refined_clean_legal_text(doc.page_content)
        

    print("Làm sạch hoàn tất.")

    # 3. Chia nhỏ văn bản (Chunking)
    print("Đang chia nhỏ tài liệu thành các chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Đã tạo {len(chunks)} chunks từ các tài liệu.")

    # 4. Tạo embedding và lưu vào FAISS
    print("Đang khởi tạo mô hình embeddings...")
    embeddings_model = GPT4AllEmbeddings(model_file=EMBEDDING_MODEL_PATH)
    
    print("Đang tạo vector store từ các chunks (quá trình này có thể mất vài phút)...")
    vector_store = FAISS.from_documents(
        chunks,
        embeddings_model
    )
    
    # 5. Lưu vector store ra đĩa
    if not os.path.exists("vector_store"):
        os.makedirs("vector_store")
    vector_store.save_local(DB_PATH)
    print(f"🎉 Vector store đã được tạo và lưu thành công tại: '{DB_PATH}'")


if __name__ == "__main__":
    build_vector_store()