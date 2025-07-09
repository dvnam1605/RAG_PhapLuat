import re
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import google.generativeai as genai
from sentence_transformers import CrossEncoder

print("Đang tải mô hình Reranker...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Tải xong mô hình Reranker.")


def rewrite_to_general_query(model: genai.GenerativeModel, query: str) -> str: 
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
---
**YÊU CẦU:**
Bây giờ, hãy viết lại câu hỏi dưới đây thành MỘT câu hỏi khái quát hơn.
**Câu hỏi gốc:** "{query}"
**Câu hỏi khái quát hơn:**
"""
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.1))
    return response.text.strip()


def decompose_query(model: genai.GenerativeModel, query: str) -> List[str]:
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


def rerank_documents_cross_encoder(
    query: str,
    documents: List[Document],
    top_n: int = 5
) -> List[Document]:
    """
    Sắp xếp lại các tài liệu bằng mô hình CrossEncoder dựa trên mức độ liên quan
    với truy vấn đầu vào.

    Args:
        query: Câu hỏi gốc của người dùng.
        documents: Danh sách các tài liệu được truy xuất ban đầu.
        top_n: Số lượng tài liệu hàng đầu cần trả về.

    Returns:
        Một danh sách các tài liệu đã được sắp xếp lại và lọc.
    """
    if not documents:
        return []

    print(f"\n🔄 Bắt đầu bước Reranking cho {len(documents)} tài liệu bằng Cross-Encoder...")

    # Tạo các cặp [câu hỏi, nội dung tài liệu] để mô hình chấm điểm
    pairs = [[query, doc.page_content] for doc in documents]

    # Dự đoán điểm số, mô hình sẽ xử lý theo batch nên rất nhanh
    scores = reranker.predict(pairs, show_progress_bar=True)

    # Gắn điểm số vào metadata của mỗi tài liệu
    for doc, score in zip(documents, scores):
        doc.metadata["rerank_score"] = float(score)

    # Sắp xếp các tài liệu dựa trên điểm số rerank (cao đến thấp)
    ranked_docs = sorted(documents, key=lambda d: d.metadata["rerank_score"], reverse=True)

    print(f"✅ Reranking hoàn tất. Trả về {min(top_n, len(ranked_docs))} tài liệu hàng đầu.")
    return ranked_docs[:top_n]


def _search_multiple_queries(
    queries: List[str],
    retriever: BaseRetriever,
) -> List[Document]:
    all_results = []
    print("---Đang thực hiện tìm kiếm cho các truy vấn con---")
    for sub_q in queries:
        print(f"  -> Đang tìm kiếm cho: '{sub_q}'")
        try:
            retrieved = retriever.invoke(sub_q)
            all_results.extend(retrieved)
        except Exception as e:
            print(f"  Lỗi khi tìm kiếm cho '{sub_q}': {e}")
            continue

    unique_docs_dict = {doc.page_content: doc for doc in all_results}
    unique_docs = list(unique_docs_dict.values())
    print(f"---Tìm thấy tổng cộng {len(all_results)} tài liệu, sau khi lọc còn {len(unique_docs)} tài liệu duy nhất.---")
    return unique_docs


def transformed_search(
    query: str,
    transformation_type: str,
    model: genai.GenerativeModel,
    retriever: BaseRetriever,
    use_reranking: bool = True,
    rerank_top_n: int = 5
) -> List[Document]:
    """
    Thực hiện tìm kiếm sử dụng truy vấn đã được biến đổi và tùy chọn áp dụng reranking
    bằng Cross-Encoder.
    """
    retrieved_docs = []

    if transformation_type == "rewrite":
        print("🔎 Bắt đầu biến đổi truy vấn: REWRITE")
        transformed_query = rewrite_to_general_query(model, query)
        print(f"  -> Câu hỏi khái quát hơn: {transformed_query}")
        retrieved_docs = retriever.invoke(transformed_query)
    elif transformation_type == "step_back" or transformation_type == "decompose":
        print(f"🔎 Bắt đầu biến đổi truy vấn: {transformation_type.upper()}")
        queries = decompose_query(model, query)
        retrieved_docs = _search_multiple_queries(queries, retriever)
    else: # 'none' hoặc bất kỳ giá trị nào khác
        print("🔎 Thực hiện tìm kiếm thông thường (không biến đổi)")
        retrieved_docs = retriever.invoke(query)

    print("\n--- DEBUG: TÀI LIỆU TRƯỚC KHI RERANK ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i+1}. {doc.page_content[:100]}...") # In 100 ký tự đầu
    print("-" * 20)

    if not use_reranking:
        print("🚫 Bỏ qua bước Reranking.")
        return retrieved_docs

    reranked_docs = rerank_documents_cross_encoder(
        query=query, # Luôn sử dụng câu hỏi GỐC để rerank
        documents=retrieved_docs,
        top_n=rerank_top_n
    )

    print("\n--- DEBUG: TÀI LIỆU SAU KHI RERANK ---")
    for i, doc in enumerate(reranked_docs):
        score = doc.metadata.get('rerank_score', 'N/A')
        print(f"{i+1}. Score: {score:.4f} - {doc.page_content[:100]}...")
    print("-" * 20)

    return reranked_docs