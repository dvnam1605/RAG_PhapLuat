
import re
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import google.generativeai as genai

 
def rewrite_to_general_query(model: genai.GenerativeModel, query: str) -> str:
    """
    Viết lại câu hỏi gốc thành một câu hỏi khác hợp lý và khái quát hơn,
    tập trung vào bản chất pháp lý của vấn đề.
    Trả về MỘT chuỗi câu hỏi.
    """
    prompt = f"""Bạn là một Trợ lý AI pháp lý chuyên nghiệp. Nhiệm vụ của bạn là nhận một câu hỏi pháp lý từ người dùng và **viết lại nó thành MỘT câu hỏi pháp lý khác, hợp lý và khái quát hơn**.
Mục tiêu là tìm ra các văn bản luật, nghị định, thông tư nền tảng liên quan đến vấn đề gốc.

**QUY TẮC:**
1.  **KHÔNG TRẢ LỜI CÂU HỎI.** Chỉ tập trung vào việc tạo ra câu hỏi mới.
2.  **KHÁI QUÁT HÓA VÀ TẬP TRUNG VÀO CỐT LÕI:** Chuyển câu hỏi chi tiết thành câu hỏi về nguyên tắc, quy định chung, hoặc bản chất pháp lý.
3.  **CHỈ MỘT CÂU HỎI.**
4.  **SỬ DỤNG NGÔN NGỮ PHÁP LÝ CHÍNH XÁC.**

---
**VÍ DỤ:**
**Câu hỏi gốc:** "Công ty nợ lương 2 tháng thì phạt thế nào?"
**Câu hỏi khái quát hơn:** "Quy định về trách nhiệm pháp lý của người sử dụng lao động đối với việc chậm trả lương cho người lao động là gì?"

**Câu hỏi gốc:** "Tôi bị hàng xóm xây nhà lấn sang 10cm đất, tôi phải làm gì?"
**Câu hỏi khái quát hơn:** "Nguyên tắc và trình tự giải quyết tranh chấp ranh giới đất đai giữa các cá nhân theo quy định pháp luật dân sự là gì?"

**Câu hỏi gốc:** "Tôi muốn ly hôn đơn phương khi chồng tôi có hành vi bạo lực gia đình và đang trốn nợ, thủ tục cần những gì và tài sản chung là một ngôi nhà sẽ được phân chia ra sao?"
**Câu hỏi khái quát hơn:** "Các căn cứ và thủ tục pháp lý cho ly hôn đơn phương, kèm theo nguyên tắc phân chia tài sản chung trong trường hợp có hành vi bạo lực gia đình và nợ xấu là gì?"
---
**YÊU CẦU:**
Bây giờ, hãy viết lại câu hỏi dưới đây thành MỘT câu hỏi khái quát hơn.

**Câu hỏi gốc:** "{query}"
**Câu hỏi khái quát hơn:**
"""
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2)) # Có thể tăng temp lên 0.2 để có câu hỏi đa dạng hơn
    return response.text.strip()


def decompose_query(model: genai.GenerativeModel, query: str) -> List[str]:
    """Phân rã câu hỏi phức tạp thành các câu hỏi con (giữ nguyên logic cũ)."""
    prompt = f"""Bạn là một chuyên gia phân tích pháp lý. Nhiệm vụ của bạn là phân rã một câu hỏi pháp lý **phức tạp** thành nhiều câu hỏi con, **đơn giản và độc lập**.
**QUY TẮC:** Mỗi câu hỏi con phải tập trung vào **MỘT** khía cạnh duy nhất và có thể trả lời độc lập. Mỗi câu hỏi con trên một dòng.
---
**VÍ DỤ:**
**Câu hỏi gốc:** "Tôi muốn ly hôn đơn phương khi chồng tôi có hành vi bạo lực gia đình và đang trốn nợ, thủ tục cần những gì và tài sản chung là một ngôi nhà sẽ được phân chia ra sao?"
**Câu hỏi con được phân rã:**
Căn cứ pháp lý để ly hôn đơn phương khi có hành vi bạo lực gia đình là gì?
Thủ tục và hồ sơ cần thiết để tiến hành ly hôn đơn phương tại Tòa án?
Nguyên tắc phân chia tài sản chung là nhà ở khi ly hôn được quy định như thế nào?
Việc một bên vợ hoặc chồng có nợ riêng ảnh hưởng thế nào đến việc phân chia tài sản chung khi ly hôn?
---
**YÊU CẦU:**
Phân rã câu hỏi phức tạp dưới đây.

**Câu hỏi gốc:** "{query}"
**Câu hỏi con được phân rã:**
"""
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))
    sub_queries = response.text.strip().split('\n')
    cleaned_queries = [re.sub(r'^\s*-\s*|\s*\d+\.\s*', '', q).strip() for q in sub_queries if q.strip()]
    return cleaned_queries if cleaned_queries else [query]



def _search_multiple_queries(
    queries: List[str],
    retriever: BaseRetriever,
    k_per_query: int = 3,
) -> List[Document]:
    """
    Hàm helper để thực hiện tìm kiếm cho một danh sách các truy vấn,
    gộp và loại bỏ các kết quả trùng lặp.
    """
    all_results = []
    print("---Đang thực hiện tìm kiếm cho các truy vấn con---")
    for sub_q in queries:
        print(f"  -> Đang tìm kiếm cho: '{sub_q}'")
        try:
            # Đảm bảo retriever hỗ trợ tham số 'k' trong invoke
            all_results.extend(retriever.invoke(sub_q, k=k_per_query))
        except Exception as e:
            print(f"  Lỗi khi tìm kiếm cho '{sub_q}': {e}")
            # Tiếp tục với truy vấn tiếp theo nếu có lỗi
            continue

    unique_docs = list({(doc.page_content, doc.metadata): doc for doc in all_results}.values()) # Sử dụng metadata để nhận diện document duy nhất tốt hơn
    print(f"---Tìm thấy tổng cộng {len(all_results)} tài liệu, sau khi lọc còn {len(unique_docs)} tài liệu duy nhất.---")
    return unique_docs


def transformed_search(query: str, transformation_type: str, model: genai.GenerativeModel, retriever: BaseRetriever) -> List[Document]:
    """Thực hiện tìm kiếm sử dụng truy vấn đã được biến đổi."""
    if transformation_type == "rewrite":
        print("🔎 Bắt đầu biến đổi truy vấn: REWRITE (Thành 1 câu khái quát)")
        transformed_query = rewrite_to_general_query(model, query)
        print(f"  -> Câu hỏi khái quát hơn: {transformed_query}")
        return retriever.invoke(transformed_query)
    elif transformation_type == "step_back" or transformation_type == "decompose": # Gộp step_back và decompose
        print(f"🔎 Bắt đầu biến đổi truy vấn: {transformation_type.upper()}")
        queries = decompose_query(model, query) # Cả hai đều có thể dùng logic phân rã
        return _search_multiple_queries(queries, retriever)
    print("🔎 Thực hiện tìm kiếm thông thường (không biến đổi)")
    return retriever.invoke(query)