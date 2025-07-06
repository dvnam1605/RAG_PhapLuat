import os
import time
from typing import TypedDict, List, Literal, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from duckduckgo_search import DDGS

from query_transform import transformed_search

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API key for Google Generative AI is not set in the environment variables.")

MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL_PATH = "model/all-MiniLM-L6-v2-f16.gguf"
VECTOR_STORE_PATH = "vector_store/faiss"
MAX_WEB_RESULTS = 5

genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel(MODEL_NAME)
embeddings = GPT4AllEmbeddings(model_file=EMBEDDING_MODEL_PATH)

try:
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(
                                            search_type='similarity',
                                            search_kwargs={'k': 15}
                                        )
    print("Hệ thống đã sẵn sàng.\n")
except Exception as e:
    print(f"LỖI: Không thể tải Vector Store. Lỗi: {e}")
    exit()


class GraphState(TypedDict):

    query: str
    result: str
    documents: List[str]
    prompt_template: str
    history: List[tuple]
    transformation_type: Optional[str]
    route_decision: str  

def handle_internal_search(state: GraphState) -> dict:

    print("---NODE: Thực hiện tìm kiếm nội bộ (FAISS)---")
    query = state['query']
    history_text = "\n".join([f"Người dùng: {q}\nTrợ lý: {a}" for q, a in state['history'][-3:]])
    context = state.get('documents', [])
    transformation_type = state.get('transformation_type', 'rewrite')
    print(f"---Sử dụng phương pháp biến đổi: {transformation_type.upper() if transformation_type else 'REGULAR'}---")
    results = transformed_search(query=query, transformation_type=transformation_type, model=llm_model, retriever=retriever)
    prompt_template = f"""
Bạn là một trợ lý pháp lý thân thiện, am hiểu văn bản pháp luật của Việt Nam.
Trả lời câu hỏi của người dùng CHỈ dựa trên phần "Dữ liệu nội bộ" dưới đây — không suy đoán hay tạo nội dung không có sẵn.

- Nếu không đủ thông tin, hãy nói rõ một cách nhã nhặn (ví dụ: "Dựa theo thông tin tôi có được...").
- Nếu câu hỏi chưa rõ ràng hoặc quá chung chung, hãy khuyến khích người dùng hỏi lại cụ thể hơn.
- Chỉ đề cập đến dữ liệu nội bộ, không nói đến cách cấu trúc hoặc cách bạn nhận được dữ liệu.
- Giải thích gọn, dễ hiểu, và đúng theo nội dung từ Bộ Tài chính, nếu cần hãy trích xuất từ ngữ cảnh được cung cấp cho đầy đủ.
- Nếu không thể xác định câu trả lời từ dữ liệu, hãy nói rõ: "Tôi không tìm thấy thông tin phù hợp trong dữ liệu nội bộ."
- Nếu người dùng yêu cầu chi tiết, hãy trả lời thật chi tiết từ nội dung nội bộ bạn được cung cấp.
- Dựa vào lịch sử trò chuyện để trả lời mạch lạc hơn nếu cần.

Lịch sử trò chuyện:
---
{history_text}
---

Dữ liệu nội bộ (từ Elasticsearch):
---
{context}
---

Câu hỏi:
{query}

Trả lời:
Hãy cung cấp một câu trả lời chi tiết, rõ ràng, và đúng theo các dữ liệu trong phần "Dữ liệu nội bộ". Nếu câu trả lời không đầy đủ, xin vui lòng làm rõ thêm với các thông tin mà bạn có được từ dữ liệu.
"""
    if not results:
        return {"documents": [], "result": "Không tìm thấy tài liệu pháp luật nào liên quan trong cơ sở dữ liệu.", "prompt_template": prompt_template}
    documents = [doc.page_content for doc in results]
    return {"documents": documents, "prompt_template": prompt_template}

def handle_web_search(state: GraphState) -> dict:
    """Node xử lý tìm kiếm web VÀ chuẩn bị prompt template, đã có xử lý Ratelimit."""
    print("---NODE: Thực hiện tìm kiếm trên web (DuckDuckGo)---")
    query = state['query']
    history_text = "\n".join([f"Người dùng: {q}\nTrợ lý: {a}" for q, a in state['history'][-3:]])
    context = state.get('documents', [])
    prompt_template = f"""
Dựa CHỦ YẾU vào các kết quả tìm kiếm sau đây để trả lời câu hỏi một cách ngắn gọn và chính xác. Trích dẫn nguồn nếu có thể.
- Dựa vào lịch sử trò chuyện để trả lời mạch lạc hơn nếu cần.
- Nếu không thể xác định câu trả lời từ dữ liệu, hãy nói rõ: "Tôi không tìm thấy thông tin phù hợp trong dữ liệu tìm kiếm trên web."
- Nếu câu hỏi chưa rõ ràng hoặc quá chung chung, hãy khuyến khích người dùng hỏi lại cụ thể hơn.
- Trả lời câu hỏi của người dùng CHỈ dựa trên phần "Kết quả tìm kiếm" dưới đây — không suy đoán hay tạo nội dung không có sẵn.

Lịch sử trò chuyện:
---
{history_text}
---

Kết quả tìm kiếm:
---
{context}
---

Câu hỏi: {query}

Trả lời:
"""
    try:
        time.sleep(2) 
        print("Đang gửi yêu cầu đến DuckDuckGo...")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_WEB_RESULTS))
        if not results:
            return {"documents": [], "result": "Không tìm thấy thông tin liên quan trên web.", "prompt_template": prompt_template}
        documents = [f"Nguồn: {r.get('href')}\nNội dung: {r.get('body')}" for r in results if r.get('body')]
        return {"documents": documents, "prompt_template": prompt_template}
    except Exception as e:
        print(f"Lỗi khi tìm kiếm trên web: {e}")
        if "Ratelimit" in str(e):
            error_message = "Hệ thống tìm kiếm trên web đang tạm thời bị giới hạn do có quá nhiều yêu cầu. Vui lòng thử lại sau vài phút."
        else:
            error_message = "Đã có lỗi xảy ra trong quá trình tìm kiếm trên Internet."
        return {"documents": [], "result": error_message, "prompt_template": prompt_template}

def generate_final_answer(state: GraphState) -> dict:
    """Node sinh câu trả lời cuối cùng."""
    print("---NODE: Sinh câu trả lời cuối cùng---")
    if not state.get('documents'):
        return {"result": state.get('result', "Lỗi: Không có tài liệu để xử lý.")}
    context = "\n\n---\n\n".join(state['documents'])
    history_text = "\n".join([f"Người dùng: {q}\nTrợ lý: {a}" for q, a in state['history'][-3:]])
    # Sử dụng state['prompt_template'] đã được tạo từ node trước
    final_prompt = state['prompt_template'].format(context=context, query=state['query'], history_text=history_text)
    response = llm_model.generate_content(final_prompt)
    return {"result": response.text.strip()}



def route_query(state: GraphState) -> dict:
    """Router là một node bình thường, nó cập nhật state với quyết định."""
    print("---ROUTER: Phân tích câu hỏi---")
    query = state['query']
    history_text = "\n".join([f"Người dùng: {q}\nTrợ lý: {a}" for q, a in state['history'][-3:]])
    routing_prompt = f"""
Câu hỏi sau đây nên được trả lời bằng cách nào?
1. Tìm kiếm thông tin chung trên web ('web_search')
2. Truy vấn cơ sở dữ liệu nội bộ về văn bản, quy định của Bộ Tài chính Việt Nam ('internal_search')

Lịch sử trò chuyện (nếu có):
---
{history_text}
---

Câu hỏi: "{query}"

Phân tích câu hỏi và lịch sử cẩn thận. Trả lời CHÍNH XÁC bằng MỘT trong hai từ: 'web_search' hoặc 'internal_search'.
Ví dụ:
- Câu hỏi "lãi suất ngân hàng hiện nay là bao nhiêu?" -> web_search
- Câu hỏi "thông tư 01/2023/TT-BTC quy định gì?" -> internal_search
- Câu hỏi "kể cho tôi một câu chuyện cười" -> web_search
"""
    response = llm_model.generate_content(routing_prompt)
    decision_text = ''.join(filter(str.isalpha, response.text.lower()))

    if "internal" in decision_text:
        print("---ROUTER: Quyết định -> Tìm kiếm nội bộ---")
        return {"route_decision": "internal_search"}
    else:
        print("---ROUTER: Quyết định -> Tìm kiếm trên web---")
        return {"route_decision": "web_search"}


def decide_next_node(state: GraphState) -> Literal["internal_search", "web_search"]:
    """Hàm điều kiện để graph biết phải đi đâu, chỉ đọc quyết định từ state."""
    return state["route_decision"]


workflow = StateGraph(GraphState)

workflow.add_node("router", route_query)
workflow.add_node("internal_search", handle_internal_search)
workflow.add_node("web_search", handle_web_search)
workflow.add_node("generate_answer", generate_final_answer)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", decide_next_node, {"internal_search": "internal_search", "web_search": "web_search"})
workflow.add_edge("internal_search", "generate_answer")
workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()
print("✅ LangGraph đã được biên dịch thành công!")


def rag_pipeline(query: str, history: List[tuple], transformation_type: Optional[str] = 'rewrite'):
    """Hàm chính để gọi và chạy LangGraph, không còn gọi router thừa."""
    inputs = {"query": query, "history": history, "transformation_type": transformation_type}
    final_state = app.invoke(inputs)
    route_decision = final_state.get('route_decision', 'Không rõ')
    final_decision_text = "Tra cứu văn bản pháp luật" if route_decision == "internal_search" else "Tìm kiếm trên Internet"
    return {"Câu hỏi": query, "Trả lời": final_state.get('result', "Lỗi."), "Phương pháp": final_decision_text}

if __name__ == "__main__":
    conversation_history = []
    while True:
        user_query = input("\n❓ Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát): ")
        if user_query.lower() in ['exit', 'quit']: break
        if not user_query.strip(): continue
        
        print("\nChọn phương pháp biến đổi truy vấn...")
        choice = input("(Enter: regular, 1: rewrite, 2: step_back, 3: decompose): ").strip()
        transform_map = {'1': 'rewrite', '2': 'step_back', '3': 'decompose'}
        selected_transform = transform_map.get(choice) # Sẽ là None nếu nhấn Enter

        result = rag_pipeline(user_query, conversation_history, transformation_type=selected_transform)
        
        print("\n" + "="*20 + " KẾT QUẢ " + "="*20)
        print(f"Câu trả lời:\n{result['Trả lời']}")
        print("=" * 50)
        conversation_history.append((user_query, result['Trả lời']))