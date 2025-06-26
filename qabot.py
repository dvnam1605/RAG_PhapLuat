from query_transform import transformed_search
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS




API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API key for Google Generative AI is not set in the environment variables.")
MODEL = "gemini-1.5-flash"
embeddings = GPT4AllEmbeddings(model_file="model/all-MiniLM-L6-v2-f16.gguf")


genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL)

path_db = "vector_store/faiss"
# Tải lại vector store từ file đã lưu
vector_store = FAISS.load_local(
    path_db, 
    embeddings,
    allow_dangerous_deserialization=True  # Add this parameter
)

number_retrievals = 10 # Số lượng văn bản cuối cùng trả về
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k': number_retrievals, # Số lượng văn bản cuối cùng trả về
        'fetch_k': 100, # Số lượng văn bản ban đầu cần lấy để chọn lọc, nên lớn hơn k
        'lambda_mult': 0.3 # Giá trị từ 0 đến 1. 0.5 là cân bằng, 1 là diversity, 0 là similarity.
    }
)


def generate_response(query, context):
    """
    Generates a response based on the query and context using Gemini.
    
    Args:
        query (str): The user query
        context (str): Contextual information to include in the response
        
    Returns:
        str: The generated response
    """
    system_prompt = (
    "Bạn là một trợ lý AI chuyên nghiệp trong việc trả lời câu hỏi thuộc lĩnh vực pháp luật, được tích hợp trong hệ thống RAG dựa trên truy xuất văn bản pháp lý. "
    "Nhiệm vụ của bạn là sử dụng **duy nhất các thông tin được cung cấp trong phần 'Bối cảnh'** để trả lời các câu hỏi pháp lý do người dùng đặt ra. "
    "Không được sử dụng kiến thức bên ngoài, không được suy luận vượt quá phạm vi thông tin có sẵn, và không được giả định hoặc phỏng đoán. "
    "Nếu phần Bối cảnh không chứa đủ dữ liệu để trả lời, bạn **phải** trả lời đúng một câu: **'Không có thông tin.'** "
    "Câu trả lời cần ngắn gọn, rõ ràng, chuẩn xác, có thể trích dẫn lại ngắn gọn quy định pháp lý nếu cần thiết, và nên thể hiện ngôn ngữ trung lập, khách quan như trong các tài liệu pháp luật."
)

    
    user_prompt = f"""
        Bạn sẽ được cung cấp một câu hỏi pháp lý từ người dùng và phần Bối cảnh chứa thông tin được truy xuất từ các văn bản luật, quy định, nghị định, hoặc án lệ.

        🔒 **Yêu cầu bắt buộc khi trả lời:**
        - Chỉ sử dụng thông tin trong phần Bối cảnh để trả lời. Không được suy luận thêm, không thêm ví dụ nếu không có trong bối cảnh.
        - Nếu bối cảnh **không chứa thông tin phù hợp** hoặc không đủ để đưa ra câu trả lời chính xác, hãy trả lời đúng một câu: **"Không có thông tin."**
        - Nếu có thể, hãy giữ lại ngôn ngữ pháp lý trung lập (như “theo quy định”, “người sử dụng lao động có nghĩa vụ”, “cơ quan có thẩm quyền…”).
        - Không cần chào hỏi hoặc giải thích dài dòng. Trả lời trực tiếp, súc tích và chính xác.

        ---

        📌 **Câu hỏi:**  
        {query}

        📚 **Bối cảnh (được truy xuất):**  
        {context}

        ---

        🧠 **Trả lời:**
        """

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0  # Giữ độ ngẫu nhiên thấp để trả lời chính xác
        )
    )
    return response.text.strip()

def rag_with_query_transformation(query, transformation_type=None):
    """
    Perform RAG with query transformation.
    
    Args:
        pdf_path (str): Path to the PDF file
        query (str): User query
        transformation_type (str): Type of transformation ('rewrite', 'step_back', or 'decompose')
        
    Returns:
        str: Generated response based on the query and context
    """
        
    if transformation_type:
        # Perform transformed search
        results = transformed_search(query, transformation_type)
    else:
        # Perform regular search
        results = mmr_retriever.invoke(query)

    context = "\n\n".join([f"PASSAGE {i+1}:\n{result.page_content}" for i, result in enumerate(results)])
    response = generate_response(query, context)

    return {
        "Câu hỏi gốc": query,
        "Dạng biến đổi": transformation_type,
        "Ngữ cảnh": context,
        "Trả lời": response
    }

if __name__ == "__main__":

    query = input("Nhập câu hỏi của bạn: ")
    transformation_type = input("Nhập dạng biến đổi (rewrite, step_back, decompose) hoặc để trống nếu không cần: ").strip() or None
    result = rag_with_query_transformation(query, transformation_type)
    print(f"Kết quả trả về: {result["Trả lời"]}")
   