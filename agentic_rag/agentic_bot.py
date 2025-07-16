import sys
import os
import uuid
import traceback
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query_transform import decompose_query, rewrite_to_general_query, rerank_documents_cross_encoder



# === 1. CÀI ĐẶT VÀ KHỞI TẠO ===
print("--- Bắt đầu quá trình cài đặt và khởi tạo ---")
load_dotenv()
API_KEY = os.getenv("API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not API_KEY: raise ValueError("API key của Google không được thiết lập.")
if not TAVILY_API_KEY: raise ValueError("API key của Tavily không được thiết lập.")

llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY, convert_system_message_to_human=True)

retriever = None
try:
    print("Đang tải Embedding Model và Vector Store...")
    # Use the correct path to the vietnamese-bi-encoder model
    model_path = os.path.join(os.path.dirname(__file__), "vietnamese-bi-encoder")
    embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    VECTOR_STORE_PATH = "vector_store/faiss"
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={'k': 20})
    print("✅ Vector Store đã sẵn sàng.")
except Exception as e:
    print(f"❌ LỖI: Không thể tải Vector Store: {e}")


# === 2. ĐỊNH NGHĨA CÁC CÔNG CỤ (TOOLS) CHO AGENT ===
@tool
def internal_search_and_rerank(query: str) -> str:
    """Công cụ QUAN TRỌNG NHẤT để tìm kiếm thông tin trong cơ sở dữ liệu pháp luật nội bộ của Việt Nam. Luôn ưu tiên sử dụng công cụ này cho các câu hỏi liên quan đến luật, nghị định, thông tư."""
    print(f"\n---TOOL: Thực hiện Tìm kiếm nội bộ & Rerank cho câu hỏi: '{query}'---")
    if not retriever: return "Lỗi: Vector store chưa được tải."
    try:
        initial_docs = retriever.invoke(query)
        if not initial_docs: return f"Không tìm thấy tài liệu nào cho câu hỏi: '{query}'"
        reranked_docs = rerank_documents_cross_encoder(query=query, documents=initial_docs, top_n=5)
        if not reranked_docs: return f"Sau khi xếp hạng lại, không có tài liệu nào phù hợp cho câu hỏi: '{query}'"
        return "\n\n---\n\n".join([f"Trích đoạn liên quan (Điểm: {doc.metadata.get('rerank_score', 0):.2f}):\n{doc.page_content}" for doc in reranked_docs])
    except Exception as e:
        traceback.print_exc()
        return f"Lỗi xảy ra trong quá trình tìm kiếm nội bộ: {e}"

@tool
def web_search(query: str) -> str:
    """Sử dụng công cụ này khi không tìm thấy thông tin trong cơ sở dữ liệu nội bộ, hoặc khi câu hỏi mang tính tổng quát, yêu cầu thông tin mới nhất."""
    print(f"\n---TOOL: Thực hiện Tìm kiếm Web (Tavily) cho câu hỏi: '{query}'---")
    try:
        search_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
        results = search_tool.invoke(query)
        if not results: return "Tìm kiếm trên web không trả về kết quả nào."
        documents = [f"Nguồn: {r.get('url')}\nNội dung: {r.get('content')}" for r in results if r.get('content')]
        if not documents: return "Tìm kiếm trên web có kết quả nhưng không chứa nội dung hữu ích."
        return "\n\n---\n\n".join(documents)
    except Exception as e:
        traceback.print_exc()
        return f"Lỗi khi tìm kiếm trên web: {e}"

@tool
def decompose_question(query: str) -> list[str]:
    """Sử dụng khi câu hỏi của người dùng phức tạp, chứa nhiều ý, nhiều vế hỏi (ví dụ: có từ 'và', 'hoặc', nhiều dấu phẩy)."""
    print(f"\n---TOOL: Phân rã câu hỏi: '{query}'---")
    return decompose_query(llm_model, query)

@tool
def rewrite_question(query: str) -> str:
    """Sử dụng khi câu hỏi của người dùng quá chi tiết, mang tính cá nhân (ví dụ: nhắc đến tên công ty) mà có thể không có trong văn bản luật chung."""
    print(f"\n---TOOL: Viết lại (khái quát hóa) câu hỏi: '{query}'---")
    return rewrite_to_general_query(llm_model, query)

tools = [internal_search_and_rerank, web_search, decompose_question, rewrite_question]


# === 3. THIẾT KẾ AGENT VÀ ĐỒ THỊ LANGGRAPH ===
print("--- Thiết kế Agent và đồ thị LangGraph ---")
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm_with_tools = llm_model.bind_tools(tools)

# SỬ DỤNG PROMPT ĐÃ NÂNG CẤP VỚI QUY TẮC TRÍCH DẪN
AGENT_PROMPT = """Bạn là một AI chuyên gia pháp lý cực kỳ thông minh và có phương pháp của Việt Nam. Bạn suy nghĩ một cách có logic, từng bước một để trả lời câu hỏi của người dùng một cách chính xác và toàn diện nhất.

**QUY TRÌNH BẮT BUỘC:**
Đối với mỗi lượt, bạn phải sử dụng một khối `Suy nghĩ:` để phân tích vấn đề và lên kế hoạch hành động. Sau đó, bạn sẽ quyết định gọi một công cụ hoặc trả lời cuối cùng.

**CÁC CÔNG CỤ BẠN CÓ:**
1.  `decompose_question(query: str)`:
    *   **KHI NÀO DÙNG:** KHI câu hỏi của người dùng phức tạp, chứa nhiều ý, nhiều vế hỏi (ví dụ: có từ 'và', 'hoặc', nhiều dấu phẩy).
    *   **TẠI SAO DÙNG:** Để chia một vấn đề lớn thành các vấn đề nhỏ, dễ quản lý. Sau khi phân rã, bạn sẽ giải quyết từng câu hỏi con.

2.  `rewrite_question(query: str)`:
    *   **KHI NÀO DÙNG:** KHI câu hỏi của người dùng quá chi tiết, mang tính cá nhân (ví dụ: nhắc đến tên công ty, một tình huống cụ thể) mà có thể không có trong văn bản luật chung.
    *   **TẠI SAO DÙNG:** Để khái quát hóa câu hỏi, giúp tìm kiếm được các quy định pháp luật gốc, nền tảng.

3.  `internal_search_and_rerank(query: str)`:
    *   **KHI NÀO DÙNG:** Đây là công cụ **ƯU TIÊN HÀNG ĐẦU** cho mọi câu hỏi liên quan đến pháp luật Việt Nam. Dùng nó để tìm kiếm trong cơ sở dữ liệu nội bộ.
    *   **TẠI SAO DÙNG:** Đây là nguồn thông tin chính xác và đáng tin cậy nhất.

4.  `web_search(query: str)`:
    *   **KHI NÀO DÙNG:** CHỈ dùng khi `internal_search_and_rerank` không tìm thấy gì, hoặc khi câu hỏi rõ ràng không phải về luật, hoặc hỏi về tin tức, sự kiện rất mới.
    *   **TẠI SAO DÙNG:** Để tìm kiếm thông tin trên Internet, như một phương án dự phòng.

---
**TRẢ LỜI CUỐI CÙNG VÀ TRÍCH DẪN NGUỒN:**
Khi bạn đã thu thập đủ thông tin từ các công cụ, hãy ngừng gọi công cụ và trả lời người dùng. Câu trả lời của bạn phải:
-   Tổng hợp, đầy đủ và dễ hiểu.
-   Trình bày rõ ràng, có cấu trúc.

**QUY TẮC TRÍCH DẪN QUAN TRỌNG:**
-   Nếu câu trả lời của bạn có sử dụng thông tin từ công cụ `web_search`, bạn **BẮT BUỘC** phải trích dẫn (cite) các nguồn (URL) mà bạn đã sử dụng.
-   Bạn có thể đặt số tham chiếu như `[1]`, `[2]` ngay sau thông tin, và liệt kê danh sách nguồn ở cuối câu trả lời.

**VÍ DỤ VỀ ĐỊNH DẠNG TỐT:**
Luật An ninh mạng 2018 của Việt Nam, có hiệu lực từ ngày 01/01/2019, bao gồm 7 chương và 41 điều. Luật này quy định về các hoạt động bảo vệ an ninh quốc gia trên không gian mạng [1].
Nguồn tham khảo:
[1] https://luatvietnam.vn/an-ninh-quoc-gia/luat-an-ninh-mang-2018-164904-d1.html

Hãy bắt đầu. Hãy nhớ luôn sử dụng khối `Suy nghĩ:` và tuân thủ quy tắc trích dẫn nguồn.
"""

def agent_node(state: AgentState):
    print("\n---NODE: AGENT ĐANG SUY NGHĨ---")
    messages_with_system_prompt = [SystemMessage(content=AGENT_PROMPT)] + state['messages']
    response = llm_with_tools.invoke(messages_with_system_prompt)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["action", "end"]:
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        print("---CONDITION: Agent đã trả lời. Kết thúc.---")
        return "end"
    else:
        print(f"---CONDITION: Agent gọi công cụ: {[tool['name'] for tool in last_message.tool_calls]}---")
        return "action"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("action", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"action": "action", "end": END})
workflow.add_edge("action", "agent")


# === 4. HÀM CHẠY PIPELINE VÀ GIAO DIỆN DÒNG LỆNH ===
def agentic_rag_pipeline(query: str, thread_id: str, app: Pregel):
    config = {"configurable": {"thread_id": thread_id}}
    final_state = app.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    return final_state['messages'][-1].content

if __name__ == "__main__":
    with SqliteSaver.from_conn_string(":memory:") as memory:
        
        app = workflow.compile(checkpointer=memory)
        print("✅ Agentic RAG Graph đã được biên dịch thành công!")

        conversation_id = str(uuid.uuid4())
        print(f"\nBắt đầu cuộc hội thoại mới. ID: {conversation_id}")
        
        while True:
            user_query = input("\n❓ Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát): ")
            if user_query.lower() in ['exit', 'quit']: break
            if not user_query.strip(): continue
                
            print("\n" + "="*20 + " AGENT ĐANG LÀM VIỆC... " + "="*20)
            answer = agentic_rag_pipeline(user_query, conversation_id, app)
            
            print("\n" + "="*20 + " KẾT QUẢ CUỐI CÙNG " + "="*20)
            print(f"Câu trả lời:\n{answer}")
            print("=" * 55)