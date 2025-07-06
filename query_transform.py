# query_transformer.py

import re
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import google.generativeai as genai



class QueryTransformer:
    """Lớp chứa các phương thức để biến đổi truy vấn người dùng."""
    def __init__(self, model_instance: genai.GenerativeModel, query_text: str):
        self.model = model_instance
        self.query = query_text

    def rewrite_query(self) -> str:
        """Viết lại và đa dạng hóa truy vấn."""
        prompt = f"""Bạn là một Trợ lý AI pháp lý chuyên nghiệp, được huấn luyện để tối ưu hóa truy vấn tìm kiếm trên cơ sở dữ liệu của Thư viện Pháp luật. Nhiệm vụ của bạn là nhận một câu hỏi pháp lý từ người dùng và viết lại nó thành nhiều truy vấn tìm kiếm **tốt hơn, chi tiết và sắc bén hơn** để hệ thống RAG có thể tìm thấy các văn bản luật, nghị định, thông tư liên quan một cách chính xác nhất.

**QUY TẮC:**
1.  **KHÔNG TRẢ LỜI CÂU HỎI.** Chỉ tập trung vào việc tạo ra các truy vấn tìm kiếm.
2.  **TẬP TRUNG VÀO TỪ KHÓA:** Chuyển câu hỏi dạng văn nói thành các cụm từ khóa pháp lý cốt lõi.
3.  **ĐA DẠNG HÓA TRUY VẤN:** Tạo ra 3-4 biến thể của truy vấn.

---
**VÍ DỤ:**
**Câu hỏi gốc:** "công ty nợ lương 2 tháng thì phạt thế nào?"
**Truy vấn được tối ưu hóa:**
- mức xử phạt doanh nghiệp chậm trả lương cho người lao động
- quy định về thời hạn thanh toán tiền lương
- trách nhiệm của người sử dụng lao động khi không trả lương đúng hạn
- khiếu nại công ty nợ lương ở đâu
---
**YÊU CẦU:**
Bây giờ, hãy tối ưu hóa câu hỏi dưới đây.

**Câu hỏi gốc:** "{self.query}"
**Truy vấn được tối ưu hóa:**
"""
        response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.1))
        return response.text.strip()
    
    def generate_step_back_query(self) -> str:
        """Tạo truy vấn khái quát hóa (step-back)."""
        prompt = f"""Bạn là một chuyên gia phân tích pháp lý. Nhiệm vụ của bạn là đọc một câu hỏi pháp lý **cụ thể** của người dùng và tạo ra một câu hỏi **khái quát hơn** (step-back).
**MỤC ĐÍCH:** Câu hỏi này dùng để tìm các nguyên tắc pháp lý chung, định nghĩa, hoặc quy định khung làm nền tảng cho vấn đề cụ thể, giúp hệ thống RAG có thêm bối cảnh.
---
**VÍ DỤ:**
**Câu hỏi gốc:** "Tôi bị hàng xóm xây nhà lấn sang 10cm đất, tôi phải làm gì?"
**Câu hỏi lùi một bước:** "Nguyên tắc pháp lý và phương thức giải quyết tranh chấp đất đai liên quan đến hành vi lấn chiếm ranh giới thửa đất là gì?"
---
**YÊU CẦU:**
Tạo câu hỏi lùi một bước cho câu hỏi dưới đây.

**Câu hỏi gốc:** "{self.query}"
**Câu hỏi lùi một bước:**"""
        response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.1))
        return response.text.strip()
    
    def decompose_query(self) -> List[str]:
        """Phân rã câu hỏi phức tạp thành các câu hỏi con."""
        prompt = f"""Bạn là một chuyên gia phân tích pháp lý. Nhiệm vụ của bạn là phân rã một câu hỏi pháp lý **phức tạp** thành nhiều câu hỏi con, **đơn giản và độc lập**.
**QUY TẮC:** Mỗi câu hỏi con phải tập trung vào **MỘT** khía cạnh duy nhất và có thể trả lời độc lập.
---
**VÍ DỤ:**
**Câu hỏi gốc:** "Tôi muốn ly hôn đơn phương khi chồng tôi có hành vi bạo lực gia đình và đang trốn nợ, thủ tục cần những gì và tài sản chung là một ngôi nhà sẽ được phân chia ra sao?"
**Câu hỏi con được phân rã:**
1. Căn cứ pháp lý để ly hôn đơn phương khi có hành vi bạo lực gia đình là gì?
2. Thủ tục và hồ sơ cần thiết để tiến hành ly hôn đơn phương tại Tòa án?
3. Nguyên tắc phân chia tài sản chung là nhà ở khi ly hôn được quy định như thế nào?
4. Việc một bên vợ hoặc chồng có nợ riêng ảnh hưởng thế nào đến việc phân chia tài sản chung khi ly hôn?
---
**YÊU CẦU:**
Phân rã câu hỏi phức tạp dưới đây.

**Câu hỏi gốc:** "{self.query}"
**Câu hỏi con được phân rã:**
"""
        response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))
        content = response.text.strip()
        subqueries = [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in content.split('\n') if re.match(r'^\s*\d+\.', line)]
        return subqueries if subqueries else [self.query]

def transformed_search(
    query: str, 
    transformation_type: str, 
    model: genai.GenerativeModel, 
    retriever: BaseRetriever,
    top_k: int = 5
) -> List[Document]:
    """
    Thực hiện tìm kiếm sử dụng truy vấn đã được biến đổi.
    Hàm này nhận model và retriever làm tham số (Dependency Injection).
    """
    query_transformer = QueryTransformer(model, query)
    
    if transformation_type == "rewrite":
        transformed_query = query_transformer.rewrite_query()
        print(f"🔎 Truy vấn đã viết lại:\n---\n{transformed_query}\n---")
        return retriever.invoke(transformed_query)
        
    elif transformation_type == "step_back":
        transformed_query = query_transformer.generate_step_back_query()
        print(f"🔎 Truy vấn 'lùi một bước': {transformed_query}")
        return retriever.invoke(transformed_query)
        
    elif transformation_type == "decompose":
        sub_queries = query_transformer.decompose_query()
        print("🔎 Truy vấn đã phân rã:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"  {i}. {sub_q}")
        
        all_results = [doc for sub_q in sub_queries for doc in retriever.invoke(sub_q)]
        
        # Lọc các tài liệu duy nhất và lấy top_k
        unique_docs = list({doc.page_content: doc for doc in all_results}.values())
        return unique_docs[:top_k]
    
    # Nếu không có transformation_type hợp lệ, thực hiện tìm kiếm thông thường
    return retriever.invoke(query)