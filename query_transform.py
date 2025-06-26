import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
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


class QueryTransformer:
    def __init__(self, model, query: str):
        self.model = model
        self.query = query

    def rewrite_query(self):
        """
        Rewrites a query to make it more specific and detailed for better retrieval using Gemini.
        
        Args:
            original_query (str): The original user query
        
        Returns:
            str: The rewritten query
        """
        # Prompt hướng dẫn giống như 'system' trong OpenAI
        system_prompt = (
        "Bạn là một trợ lý AI chuyên gia trong việc cải thiện truy vấn tìm kiếm thông tin trong lĩnh vực pháp luật. "
        "Nhiệm vụ của bạn là viết lại truy vấn đầu vào của người dùng sao cho cụ thể hơn, chi tiết hơn, rõ ràng hơn về mặt pháp lý. "
        "Truy vấn mới nên bổ sung các khái niệm pháp lý, điều khoản luật, mốc thời gian, hành vi vi phạm hoặc chủ thể liên quan, "
        "nhằm tăng khả năng truy xuất đúng thông tin từ cơ sở tri thức pháp luật. "
        "Mục tiêu là tạo ra một truy vấn sắc nét, rõ ràng và chứa đủ ngữ cảnh để hệ thống hiểu chính xác mục đích tìm kiếm."
    )


        
        # Nội dung yêu cầu
        user_prompt = f"""
    Hãy viết lại truy vấn pháp lý dưới đây sao cho cụ thể và chi tiết hơn. 
    Truy vấn mới cần bổ sung:
    - Các thuật ngữ pháp lý chuyên ngành (nếu có).
    - Các yếu tố như hành vi, đối tượng, khung thời gian, loại quan hệ pháp luật hoặc điều khoản luật có thể áp dụng.
    - Các tình huống tương đương hoặc quy định tương ứng trong luật hiện hành.

    Mục tiêu là để truy vấn được rõ ràng, chính xác hơn và dễ dàng tìm thấy câu trả lời phù hợp từ hệ thống pháp luật.

    🔍 Truy vấn gốc: {self.query}

    ✏️ Truy vấn đã được viết lại:
    """



        # Gộp system + user prompt thành 1 đoạn duy nhất vì Gemini không tách "system"/"user"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Gọi model sinh ra kết quả
        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
            temperature=0.0
            )
        )
        
        # Trả kết quả sau khi strip
        return response.text.strip()
    
    def generate_step_back_query(self):
        """
        Generates a more general 'step-back' query to retrieve broader context.
        
        Args:
            original_query (str): The original user query
            model (str): The model to use for step-back query generation
            
        Returns:
            str: The step-back query
        """
        # Define the system prompt to guide the AI assistant's behavior
        system_prompt = (
        "Bạn là một trợ lý AI chuyên gia trong việc mở rộng và cải thiện các truy vấn tìm kiếm thông tin pháp luật. "
        "Nhiệm vụ của bạn là nhận truy vấn đầu vào từ người dùng và tạo ra một phiên bản truy vấn mang tính khái quát, mở rộng hơn — còn gọi là 'step-back query'. "
        "Truy vấn mở rộng cần bao gồm các yếu tố pháp lý liên quan như: loại tranh chấp, luật áp dụng, bối cảnh pháp lý, thời điểm xảy ra, cơ quan có thẩm quyền, và các điều khoản luật tiềm năng. "
        "Truy vấn mới nên giúp truy xuất được thông tin nền, án lệ, nguyên tắc pháp luật hoặc quy định có liên quan, nhằm cung cấp bối cảnh đầy đủ hơn để hỗ trợ truy vấn ban đầu trong hệ thống RAG."
        )

        # Define the user prompt with the original query to be generalized
        user_prompt = f"""
    Truy vấn sau đây được đưa ra bởi người dùng liên quan đến một vấn đề pháp lý cụ thể. 
    Nhiệm vụ của bạn là mở rộng truy vấn này thành một truy vấn 'step-back' — mang tính tổng quát và khái quát hơn, nhưng vẫn giữ được ngữ cảnh pháp lý cốt lõi. 

    Truy vấn mở rộng cần hướng đến việc:
    - Bao phủ các khía cạnh pháp luật liên quan như: điều luật, phạm vi áp dụng, khái niệm pháp lý chung, hoặc các loại tranh chấp tương tự.
    - Làm rõ bối cảnh chung, các khung pháp lý, cơ sở pháp lý có thể áp dụng cho truy vấn gốc.
    - Tạo điều kiện để hệ thống có thể truy xuất được thêm thông tin hỗ trợ, làm rõ hoặc cung cấp nền tảng giải thích cho truy vấn cụ thể.

    **Truy vấn gốc (original query)**: {self.query}

    🔍 **Truy vấn mở rộng (step-back query)**:
        """


        
        # Generate the step-back query using the specified model
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Gọi model sinh ra kết quả
        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
            temperature=0.1 
            )
        )
        
        # Trả kết quả sau khi strip
        return response.text.strip()
    
    def decompose_query(self, num_subqueries=4):
        """
        Decomposes a complex query into simpler sub-queries.
        
        Args:
            original_query (str): The original user query
        
        Returns:
            list: A list of decomposed sub-queries
        """
        # Define the system prompt to guide the AI assistant's behavior
        system_prompt = (
        "Bạn là một trợ lý AI chuyên gia trong việc phân tích và phân rã truy vấn tìm kiếm trong lĩnh vực pháp luật. "
        "Nhiệm vụ của bạn là nhận một truy vấn phức tạp từ người dùng — thường chứa nhiều khía cạnh pháp lý khác nhau — "
        "và phân tách truy vấn này thành các truy vấn con đơn giản, mỗi truy vấn tập trung vào một yếu tố hoặc vấn đề pháp lý cụ thể. "
        "Điều này sẽ giúp hệ thống tìm kiếm hoạt động hiệu quả hơn, truy xuất được thông tin chính xác hơn, "
        "đặc biệt trong các hệ thống RAG sử dụng cơ sở tri thức luật hoặc văn bản pháp quy."
    )

        
        # Define the user prompt with the original query to be decomposed
        user_prompt = f"""
    Truy vấn sau đây chứa nhiều khía cạnh pháp lý khác nhau, cần được phân tách để hệ thống có thể tìm kiếm chính xác hơn. 
    Hãy chia nhỏ truy vấn phức tạp này thành {num_subqueries} truy vấn con, mỗi truy vấn nên tập trung vào **một khía cạnh riêng biệt**, 
    chẳng hạn như: hành vi vi phạm, chủ thể liên quan, căn cứ pháp lý, quy trình xử lý, hoặc khung hình phạt áp dụng.

    Yêu cầu định dạng:
    1. [Truy vấn con 1]
    2. [Truy vấn con 2]
    ...

    🔍 Truy vấn gốc: {self.query}
    """


        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2  # Giữ độ ngẫu nhiên thấp để truy vấn con có tính chính xác cao
            )
        )
        content = response.text.strip()
        # print(f"Nội dung trả về từ model: {content}")
        lines = content.split('\n')
        # print(f"Có {len(lines)} dòng trong nội dung trả về từ model.")
        
        subqueries = []
        for line in lines:
            print(f"Đang xử lý dòng: {line.strip()}")
            if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
                query = line.strip()
                query = query[query.find('.') + 1:].strip()  # Loại bỏ số thứ tự
                subqueries.append(query)
        return subqueries  # Trả về đúng số lượng truy vấn con yêu cầu

def transformed_search(query, transformation_type, top_k=3):
    """
    Search using a transformed query.
    
    Args:
        query (str): Original query
        vector_store (SimpleVectorStore): Vector store to search
        transformation_type (str): Type of transformation ('rewrite', 'step_back', or 'decompose')
        top_k (int): Number of results to return
        
    Returns:
        List[Dict]: Search results
    """
    print(f"Transformation type: {transformation_type}")
    print(f"Original query: {query}")
    
    results = []

    query_transformer = QueryTransformer(model, query)
    
    if transformation_type == "rewrite":
        # Query rewriting
        transformed_query = query_transformer.rewrite_query()
        print(f"Rewritten query: {transformed_query}")
        
        # FIX: Pass the transformed query string directly to the retriever.
        # The retriever will handle the embedding.
        results = mmr_retriever.invoke(transformed_query)
        
    elif transformation_type == "step_back":
        # Step-back prompting
        transformed_query = query_transformer.generate_step_back_query()
        print(f"Step-back query: {transformed_query}")
        
        results = mmr_retriever.invoke(transformed_query)
        
    elif transformation_type == "decompose":
        # Sub-query decomposition
        sub_queries = query_transformer.decompose_query(num_subqueries= top_k)
        print("Decomposed into sub-queries:")
        
        # Search with each sub-query and combine results
        all_results = []
        for sub_q in sub_queries:
            sub_results = mmr_retriever.invoke(sub_q)
            all_results.extend(sub_results)
        
        # 2. Lọc ra các tài liệu duy nhất (unique)
        unique_docs = []
        seen_contents = set()  # Dùng set để kiểm tra trùng lặp hiệu quả
        for doc in all_results:
            # Chỉ thêm tài liệu vào kết quả nếu nội dung của nó chưa từng xuất hiện
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        # 3. Lấy top_k từ danh sách các tài liệu đã được lọc duy nhất
        results = unique_docs[:top_k]
        
    else:
        # FIX: For regular search, pass the original query string directly.
        results = mmr_retriever.invoke(query)
    
    return results