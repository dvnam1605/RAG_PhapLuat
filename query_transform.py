import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import re # Thêm thư viện re để xử lý chuỗi tốt hơn

# --- 1. CẤU HÌNH VÀ TẢI MÔ HÌNH ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API key for Google Generative AI is not set in the environment variables.")

# Cấu hình mô hình và embeddings
MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL_PATH = "model/all-MiniLM-L6-v2-f16.gguf"
VECTOR_STORE_PATH = "vector_store/faiss"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)
embeddings = GPT4AllEmbeddings(model_file=EMBEDDING_MODEL_PATH)

# Tải lại vector store từ file đã lưu
print(f"Đang tải Vector Store từ: {VECTOR_STORE_PATH}...")
try:
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Tải Vector Store thành công.")
except Exception as e:
    print(f"Lỗi khi tải Vector Store: {e}")
    print("Vui lòng đảm bảo bạn đã tạo và lưu trữ Vector Store đúng đường dẫn.")
    exit()


# --- 2. CẤU HÌNH RETRIEVER ---
number_retrievals = 10 # Số lượng văn bản cuối cùng trả về
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k': number_retrievals, # Số lượng văn bản cuối cùng trả về
        'fetch_k': 50, # Số lượng văn bản ban đầu cần lấy để chọn lọc, nên lớn hơn k
        'lambda_mult': 0.4 # Giá trị từ 0 đến 1. 0.5 là cân bằng, 1 là diversity, 0 là similarity.
    }
)


# --- 3. LỚP BIẾN ĐỔI TRUY VẤN (QUERY TRANSFORMER) VỚI PROMPT ĐÃ TỐI ƯU ---
class QueryTransformer:
    def __init__(self, model, query: str):
        self.model = model
        self.query = query

    def rewrite_query(self):
        """
        Viết lại và đa dạng hóa truy vấn gốc thành các truy vấn tìm kiếm pháp lý hiệu quả hơn.
        Sử dụng kỹ thuật few-shot prompting để hướng dẫn mô hình.
        """
        prompt = f"""
Bạn là một Trợ lý AI pháp lý chuyên nghiệp, được huấn luyện để tối ưu hóa truy vấn tìm kiếm trên cơ sở dữ liệu của Thư viện Pháp luật.
Nhiệm vụ của bạn là nhận một câu hỏi pháp lý từ người dùng và viết lại nó thành nhiều truy vấn tìm kiếm **tốt hơn, chi tiết và sắc bén hơn** để hệ thống RAG có thể tìm thấy các văn bản luật, nghị định, thông tư liên quan một cách chính xác nhất.

**QUY TẮC:**
1.  **KHÔNG TRẢ LỜI CÂU HỎI.** Chỉ tập trung vào việc tạo ra các truy vấn tìm kiếm.
2.  **TẬP TRUNG VÀO TỪ KHÓA:** Chuyển câu hỏi dạng văn nói thành các cụm từ khóa pháp lý cốt lõi.
3.  **BỔ SUNG THUẬT NGỮ PHÁP LÝ:** Thêm các thuật ngữ chuyên ngành liên quan (ví dụ: "sổ đỏ" -> "giấy chứng nhận quyền sử dụng đất", "tiền đền bù" -> "tiền bồi thường khi nhà nước thu hồi đất").
4.  **ĐA DẠNG HÓA TRUY VẤN:** Tạo ra 3-4 biến thể của truy vấn, mỗi biến thể nhìn vào một khía cạnh hoặc sử dụng từ đồng nghĩa khác nhau để tăng khả năng bao phủ.
5.  **GIỮ NGUYÊN Ý ĐỊNH:** Các truy vấn mới phải giữ đúng ý định tìm kiếm của câu hỏi gốc.

---
**VÍ DỤ 1:**
**Câu hỏi gốc:** "Làm sổ đỏ lần đầu hết bao nhiêu tiền?"
**Truy vấn được tối ưu hóa:**
- chi phí làm giấy chứng nhận quyền sử dụng đất lần đầu
- nghĩa vụ tài chính khi cấp sổ đỏ lần đầu
- lệ phí trước bạ và phí thẩm định hồ sơ cấp giấy chứng nhận quyền sử dụng đất
- thủ tục và các loại thuế phí khi xin cấp sổ đỏ lần đầu

**VÍ DỤ 2:**
**Câu hỏi gốc:** "công ty nợ lương 2 tháng thì phạt thế nào?"
**Truy vấn được tối ưu hóa:**
- mức xử phạt doanh nghiệp chậm trả lương cho người lao động
- quy định về thời hạn thanh toán tiền lương
- trách nhiệm của người sử dụng lao động khi không trả lương đúng hạn
- khiếu nại công ty nợ lương ở đâu

---
**YÊU CẦU:**
Bây giờ, hãy áp dụng các quy tắc và ví dụ trên để tối ưu hóa câu hỏi dưới đây.

**Câu hỏi gốc:** "{self.query}"
**Truy vấn được tối ưu hóa:**
"""

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1 # Nhiệt độ thấp để bám sát vào ví dụ và chỉ dẫn
            )
        )
        return response.text.strip()
    
    def generate_step_back_query(self):
        """
        Tạo ra một truy vấn 'lùi một bước' (step-back) để tìm kiếm bối cảnh và nguyên tắc pháp lý chung.
        """
        prompt = f"""
Bạn là một chuyên gia phân tích pháp lý. Nhiệm vụ của bạn là đọc một câu hỏi pháp lý **cụ thể** của người dùng và tạo ra một câu hỏi **khái quát hơn**, hay còn gọi là câu hỏi "lùi một bước" (step-back).

**MỤC ĐÍCH:** Câu hỏi "lùi một bước" dùng để tìm các nguyên tắc pháp lý chung, định nghĩa, hoặc quy định khung làm nền tảng cho vấn đề cụ thể mà người dùng đang hỏi. Nó giúp hệ thống RAG có thêm bối cảnh để trả lời tốt hơn.

**QUY TẮC:**
1.  **KHÔNG TRẢ LỜI CÂU HỎI GỐC.**
2.  **KHÁI QUÁT HÓA:** Chuyển từ trường hợp cụ thể sang vấn đề pháp lý chung.
3.  **TẬP TRUNG VÀO NGUYÊN TẮC:** Câu hỏi lùi bước nên hỏi về "nguyên tắc", "quy định chung", "khái niệm", "thẩm quyền", "căn cứ pháp lý".

---
**VÍ DỤ 1:**
**Câu hỏi gốc:** "Công ty tôi ở Quận 1, TPHCM, nợ lương nhân viên 2 tháng thì bị phạt thế nào theo nghị định 12/2022?"
**Câu hỏi lùi một bước:** "Quy định chung của pháp luật lao động về nghĩa vụ trả lương và các hình thức xử phạt khi doanh nghiệp vi phạm nghĩa vụ trả lương cho người lao động là gì?"

**VÍ DỤ 2:**
**Câu hỏi gốc:** "Tôi bị hàng xóm xây nhà lấn sang 10cm đất, tôi phải làm gì?"
**Câu hỏi lùi một bước:** "Nguyên tắc pháp lý và phương thức giải quyết tranh chấp đất đai liên quan đến hành vi lấn chiếm ranh giới thửa đất là gì?"

---
**YÊU CẦU:**
Bây giờ, hãy tạo câu hỏi lùi một bước cho câu hỏi dưới đây.

**Câu hỏi gốc:** "{self.query}"
**Câu hỏi lùi một bước:**
"""

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1
            )
        )
        return response.text.strip()
    
    def decompose_query(self):
        """
        Phân rã một câu hỏi phức tạp thành các câu hỏi con độc lập.
        AI sẽ tự quyết định số lượng câu hỏi con phù hợp.
        """
        prompt = f"""
Bạn là một chuyên gia phân tích pháp lý. Nhiệm vụ của bạn là phân rã một câu hỏi pháp lý **phức tạp** (chứa nhiều vấn đề) thành nhiều câu hỏi con, **đơn giản và độc lập**.

**MỤC ĐÍCH:** Việc phân rã giúp hệ thống RAG tìm kiếm thông tin chính xác cho từng khía cạnh của vấn đề, sau đó tổng hợp lại để có câu trả lời đầy đủ.

**QUY TẮC:**
1.  **ĐỘC LẬP:** Mỗi câu hỏi con phải tập trung vào **MỘT** khía cạnh duy nhất và có thể được trả lời độc lập.
2.  **BẢO TOÀN THÔNG TIN:** Tổng hợp các câu hỏi con phải bao quát được toàn bộ ý định của câu hỏi gốc.
3.  **TỰ QUYẾT ĐỊNH SỐ LƯỢNG:** Phân rã thành số lượng câu hỏi con phù hợp với độ phức tạp của câu hỏi gốc (thường từ 2 đến 5 câu). Nếu câu hỏi gốc đã đủ đơn giản, chỉ cần trả về chính nó.
4.  **GIỮ ĐỊNH DẠNG:** Xuất kết quả dưới dạng danh sách có đánh số.

---
**VÍ DỤ 1:**
**Câu hỏi gốc:** "Tôi muốn ly hôn đơn phương khi chồng tôi có hành vi bạo lực gia đình và đang trốn nợ, thủ tục cần những gì và tài sản chung là một ngôi nhà sẽ được phân chia ra sao?"
**Câu hỏi con được phân rã:**
1. Căn cứ pháp lý để ly hôn đơn phương khi có hành vi bạo lực gia đình là gì?
2. Thủ tục và hồ sơ cần thiết để tiến hành ly hôn đơn phương tại Tòa án?
3. Nguyên tắc phân chia tài sản chung là nhà ở khi ly hôn được quy định như thế nào?
4. Việc một bên vợ hoặc chồng có nợ riêng ảnh hưởng thế nào đến việc phân chia tài sản chung khi ly hôn?

**VÍ DỤ 2:**
**Câu hỏi gốc:** "Mức phạt nồng độ cồn khi lái xe máy năm 2024?"
**Câu hỏi con được phân rã:**
1. Mức xử phạt vi phạm hành chính đối với hành vi điều khiển xe máy có nồng độ cồn trong máu hoặc hơi thở năm 2024?

---
**YÊU CẦU:**
Bây giờ, hãy phân rã câu hỏi phức tạp dưới đây.

**Câu hỏi gốc:** "{self.query}"
**Câu hỏi con được phân rã:**
"""

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2
            )
        )
        content = response.text.strip()
        
        # Cải thiện logic tách câu hỏi con
        lines = content.split('\n')
        subqueries = []
        for line in lines:
            # Dùng regex để tìm các dòng bắt đầu bằng số và dấu chấm
            match = re.match(r'^\s*\d+\.\s*(.*)', line)
            if match:
                subqueries.append(match.group(1).strip())
        
        # Nếu không phân rã được, trả về câu hỏi gốc trong một list
        if not subqueries:
            return [self.query]
        return subqueries


# --- 4. HÀM TÌM KIẾM ĐÃ BIẾN ĐỔI ---
def transformed_search(query, transformation_type, top_k=5):
    """
    Thực hiện tìm kiếm sử dụng truy vấn đã được biến đổi.
    
    Args:
        query (str): Truy vấn gốc từ người dùng.
        transformation_type (str): Loại biến đổi ('rewrite', 'step_back', 'decompose', hoặc 'regular').
        top_k (int): Số lượng kết quả cuối cùng cần trả về.
        
    Returns:
        List[Document]: Danh sách các tài liệu được tìm thấy.
    """
    print(f"\n{'='*20} BẮT ĐẦU TÌM KIẾM {'='*20}")
    print(f"Loại biến đổi: {transformation_type.upper()}")
    print(f"Truy vấn gốc: {query}")
    
    results = []
    query_transformer = QueryTransformer(model, query)
    
    if transformation_type == "rewrite":
        transformed_query = query_transformer.rewrite_query()
        print(f"Truy vấn đã viết lại (đưa vào retriever):\n---\n{transformed_query}\n---")
        results = mmr_retriever.invoke(transformed_query)
        
    elif transformation_type == "step_back":
        transformed_query = query_transformer.generate_step_back_query()
        print(f"Truy vấn 'lùi một bước': {transformed_query}")
        results = mmr_retriever.invoke(transformed_query)
        
    elif transformation_type == "decompose":
        # Cập nhật: không cần truyền num_subqueries nữa
        sub_queries = query_transformer.decompose_query()
        print("Truy vấn đã phân rã:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"  {i}. {sub_q}")
        
        all_results = []
        print("\nThực hiện tìm kiếm cho từng truy vấn con...")
        for sub_q in sub_queries:
            sub_results = mmr_retriever.invoke(sub_q)
            all_results.extend(sub_results)
        
        # Lọc các tài liệu duy nhất và lấy top_k
        unique_docs = []
        seen_contents = set()
        for doc in all_results:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        results = unique_docs[:top_k]
        
    else: # 'regular' search
        print("Thực hiện tìm kiếm thông thường (không biến đổi).")
        results = mmr_retriever.invoke(query)
    
    print(f"Đã tìm thấy {len(results)} tài liệu.")
    print(f"{'='*20} KẾT THÚC TÌM KIẾM {'='*20}\n")
    return results


# --- 5. HÀM MAIN ĐỂ KIỂM TRA ---
if __name__ == "__main__":
    # Câu hỏi phức tạp để kiểm tra
    complex_query = "Tôi muốn mở một cửa hàng bán lẻ mỹ phẩm tại Hà Nội, tôi cần chuẩn bị những giấy tờ đăng ký kinh doanh gì và các loại thuế nào tôi sẽ phải đóng hàng năm?"
    
    # Câu hỏi đơn giản
    simple_query = "thủ tục đăng ký tạm trú cho người nước ngoài"

    # --- Chạy thử nghiệm với câu hỏi phức tạp ---
    print(f"--- THỬ NGHIỆM VỚI CÂU HỎI PHỨC TẠP ---")
    
    # 1. Tìm kiếm thông thường
    regular_results = transformed_search(complex_query, 'regular')
    
    # 2. Tìm kiếm với Rewrite
    rewrite_results = transformed_search(complex_query, 'rewrite')
    
    # 3. Tìm kiếm với Step-back
    step_back_results = transformed_search(complex_query, 'step_back')
    
    # 4. Tìm kiếm với Decompose
    decompose_results = transformed_search(complex_query, 'decompose', top_k=5)
    
    print("\n--- KẾT QUẢ TÌM KIẾM (DECOMPOSE) ---")
    if decompose_results:
        for i, doc in enumerate(decompose_results, 1):
            print(f"  [{i}] Nguồn: {doc.metadata.get('source', 'N/A')}")
            # print(f"  Nội dung: {doc.page_content[:300]}...\n") # In 300 ký tự đầu
    else:
        print("  Không tìm thấy tài liệu nào.")