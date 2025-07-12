import os
import time
import traceback
from typing import TypedDict, List, Literal, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from query_transform import transformed_search
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
API_KEY = os.getenv("API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not API_KEY:
    raise ValueError("API key for Google Generative AI is not set in the environment variables.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the environment variables.")

MODEL_NAME = "gemini-1.5-flash"
VECTOR_STORE_PATH = "vector_store/faiss"
MAX_WEB_RESULTS = 5

print("Đang cấu hình các mô hình...")
genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel(MODEL_NAME)
model_path = "models/vietnamese-bi-encoder"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

try:
    print(f"Đang tải Vector Store từ '{VECTOR_STORE_PATH}'...")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 20}
    )
    print("✅ Hệ thống đã sẵn sàng.\n")
except Exception as e:
    print(f"❌ LỖI: Không thể tải Vector Store. Vui lòng chạy chức năng 'build' trước.")
    print(f"   Lỗi chi tiết: {e}")
    retriever = None

class GraphState(TypedDict):
    query: str
    result: str
    documents: List[str]
    prompt_template: str
    history: List[tuple]
    transformation_type: Optional[str]
    route_decision: str
    web_search_failed: bool = False

def handle_internal_search(state: GraphState) -> dict:
    if state.get('web_search_failed', False):
        print("---NODE: Tìm kiếm web thất bại, chuyển sang tìm kiếm nội bộ (FAISS)---")
    else:
        print("---NODE: Thực hiện tìm kiếm nội bộ (FAISS)---")

    query = state['query']
    transformation_type = state.get('transformation_type', None)
    print(f"---Sử dụng phương pháp biến đổi: {transformation_type.upper() if transformation_type else 'REGULAR'}---")
    results = transformed_search(query=query, transformation_type=transformation_type, model=llm_model, retriever=retriever, rerank_top_n=5)
    
    documents = [doc.page_content for doc in results]
    prompt_template = f"""Bạn là một trợ lý pháp lý AI thông minh và trung thực. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng CHỈ dựa vào phần "Dữ liệu nội bộ" được cung cấp. Luôn trung thực, nếu không có thông tin, hãy nói rõ là không có.
**Dữ liệu nội bộ:**
---
{{context}}
---
**Câu hỏi:** {query}
**Trả lời:**
"""
    return {"documents": documents, "prompt_template": prompt_template, "web_search_failed": False}

def handle_web_search(state: GraphState) -> dict:
    print("---NODE: Thực hiện tìm kiếm trên web (Tavily)---")
    query = state['query']
    prompt_template = f"""Dựa CHỦ YẾU vào các kết quả tìm kiếm sau đây để trả lời câu hỏi một cách chi tiết và chính xác. Trích dẫn tất cả các nguồn.
**Kết quả tìm kiếm:**
---
{{context}}
---
**Câu hỏi:** {query}
**Trả lời:**
"""
    search_tool = TavilySearchResults(max_results=MAX_WEB_RESULTS, api_key=TAVILY_API_KEY )
    
    try:
        print(f"Đang gửi yêu cầu đến Tavily với câu hỏi: '{query}'")
        results = search_tool.invoke(query)


        if isinstance(results, list) and results:
            documents = []
            for r in results:
                # Lấy nội dung, ưu tiên key 'content'. Nếu không có, thử key 'body'.
                content = r.get('content') or r.get('body') 
                url = r.get('url')
                
                if content and url:
                    documents.append(f"Nguồn: {url}\nNội dung: {content}")

            if documents:
                print(f"--- Xử lý thành công {len(documents)} kết quả từ Tavily. ---")
                return {"documents": documents, "prompt_template": prompt_template, "web_search_failed": False}

        print("--- Tìm kiếm web không trả về nội dung hợp lệ (danh sách rỗng hoặc không có content/url). Coi như thất bại. ---")
        return {"web_search_failed": True, "documents": []}

    except Exception as e:
        print(f"!!! LỖI khi thực thi tìm kiếm trên web !!!")
        traceback.print_exc()
        return {"web_search_failed": True, "documents": []}

def generate_final_answer(state: GraphState) -> dict:
    print("---NODE: Sinh câu trả lời cuối cùng---")
    if not state.get('documents'):
        result = "Rất tiếc, tôi không tìm thấy thông tin phù hợp trong cả cơ sở dữ liệu nội bộ và trên web để trả lời câu hỏi của bạn."
        return {"result": result}
        
    context = "\n\n---\n\n".join(state['documents'])
    final_prompt = state['prompt_template'].format(context=context, query=state['query'])
    response = llm_model.generate_content(final_prompt)
    return {"result": response.text.strip()}

def final_router(state: GraphState) -> dict:
    print("---ROUTER (FINAL): Thực thi định tuyến dựa trên mệnh lệnh---")
    query = state['query']
    try:
        relevant_docs = retriever.invoke(query, k=1)
        evidence = relevant_docs[0].page_content if relevant_docs else "Không tìm thấy tài liệu nào trong cơ sở dữ liệu nội bộ."
    except Exception as e:
        evidence = f"Lỗi khi truy vấn cơ sở dữ liệu nội bộ: {e}"

    routing_prompt = f"""**MỆNH LỆNH HỆ THỐNG:** BẠN LÀ MỘT BỘ ĐỊNH TUYẾN. NHIỆM VỤ CỦA BẠN LÀ ƯU TIÊN SỬ DỤNG CƠ SỞ DỮ LIỆU NỘI BỘ.
**QUY TẮC:**
1.  **PHÂN TÍCH BẰNG CHỨNG:** Xem xét kỹ phần `BẰNG CHỨNG TỪ CSDL NỘI BỘ` dưới đây.
2.  **RA QUYẾT ĐỊNH:**
    *   Nếu `BẰNG CHỨNG` có chứa thông tin liên quan đến `CÂU HỎI`, BẠN **BẮT BUỘC** PHẢI chọn `internal_search`.
    *   Bạn CHỈ được phép chọn `web_search` khi `BẰNG CHỨNG` hoàn toàn không liên quan.
**KHI CÓ BẤT KỲ NGHI NGỜ NÀO, LUÔN MẶC ĐỊNH CHỌN `web_search`.**
---
**BẰNG CHỨNG TỪ CSDL NỘI BỘ:**
{evidence}
---
**CÂU HỎI CỦA NGƯỜI DÙNG:**
"{query}"
---
**QUYẾT ĐỊNH CỦA BẠN (CHỈ MỘT TỪ):**
"""
    response = llm_model.generate_content(routing_prompt)
    decision_text = response.text.strip().lower()
    if "internal_search" in decision_text:
        print("---ROUTER (FINAL): Quyết định -> Tìm kiếm nội bộ (internal_search)---")
        return {"route_decision": "internal_search"}
    else:
        print("---ROUTER (FINAL): Quyết định -> Tìm kiếm trên web (web_search)---")
        return {"route_decision": "web_search"}


def decide_initial_route(state: GraphState) -> Literal["internal_search", "web_search"]:
    """Quyết định hướng đi ban đầu."""
    return state["route_decision"]

def decide_after_web_search(state: GraphState) -> Literal["generate_answer", "internal_search"]:
    """Quyết định sau khi tìm kiếm web, thực hiện logic fallback."""
    if state.get('web_search_failed', False):
        print("---CONDITION: Tìm kiếm web thất bại, chuyển hướng sang internal_search.---")
        return "internal_search"
    else:
        print("---CONDITION: Tìm kiếm web thành công, chuyển hướng sang generate_answer.---")
        return "generate_answer"

print("Đang xây dựng đồ thị LangGraph...")
workflow = StateGraph(GraphState)

workflow.add_node("router", final_router)
workflow.add_node("internal_search", handle_internal_search)
workflow.add_node("web_search", handle_web_search)
workflow.add_node("generate_answer", generate_final_answer)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", decide_initial_route, {"internal_search": "internal_search", "web_search": "web_search"})
workflow.add_conditional_edges("web_search", decide_after_web_search, {"generate_answer": "generate_answer", "internal_search": "internal_search"})
workflow.add_edge("internal_search", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()
print("✅ LangGraph với logic fallback đã được biên dịch thành công!")


def rag_pipeline(query: str, history: List[tuple], transformation_type: Optional[str]):
    """Hàm chính để gọi và chạy LangGraph."""
    if not retriever:
        return {"Câu hỏi": query, "Trả lời": "Lỗi: Vector store chưa được tải. Không thể thực hiện truy vấn.", "Phương pháp": "Lỗi hệ thống"}
    inputs = {"query": query, "history": history, "transformation_type": transformation_type}
    final_state = app.invoke(inputs)
    route_decision = final_state.get('route_decision', 'Không rõ')
    final_decision_text = "Tra cứu văn bản pháp luật" if route_decision == "internal_search" else "Tìm kiếm trên Internet"
    return {"Câu hỏi": query, "Trả lời": final_state.get('result', "Lỗi."), "Phương pháp": final_decision_text}

if __name__ == "__main__":
    conversation_history = []
    while True:
        user_query = input("\n❓ Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát): ")
        if user_query.lower() in ['exit', 'quit']: 
            break
        if not user_query.strip(): 
            continue
        
        print("\nChọn phương pháp biến đổi truy vấn...")
        choice = input("(Enter: regular, 1: rewrite, 2: decompose): ").strip()
        transform_map = {'1': 'rewrite', '2': 'decompose'}
        selected_transform = transform_map.get(choice)

        result = rag_pipeline(user_query, conversation_history, transformation_type=selected_transform)
        
        print("\n" + "="*20 + " KẾT QUẢ " + "="*20)
        print(f"Câu trả lời:\n{result['Trả lời']}")
        print(f"(Phương pháp: {result['Phương pháp']})")
        print("=" * 50)
        conversation_history.append((user_query, result['Trả lời']))