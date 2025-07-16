# filename: app.py

import streamlit as st
import uuid
import sqlite3
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


try:
    import agentic_bot
except ImportError as e:
    st.error(f"Lỗi: Không thể import module `agentic_bot`. Hãy đảm bảo file `agentic_bot.py` tồn tại trong cùng thư mục. Lỗi chi tiết: {e}")
    st.stop()

load_dotenv()

@st.cache_resource
def load_and_compile_agentic_app():
    """
    Hàm này khởi tạo kết nối DB, tạo checkpointer và biên dịch agent.
    Nó được cache bởi Streamlit để chỉ chạy một lần.
    """
    # st.info("Lần đầu khởi tạo, đang tải các mô hình và thiết lập Trợ lý AI... Vui lòng chờ một lát.")

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    
    memory = agentic_bot.SqliteSaver(conn=conn)
    
    compiled_app = agentic_bot.workflow.compile(checkpointer=memory)
    
    # st.success("Trợ lý AI đã sẵn sàng!")
    return compiled_app

app = load_and_compile_agentic_app()


st.set_page_config(
    page_title="Trợ lý Pháp luật",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(to bottom right, #f2f6fc, #e8ecf4); }
    [data-testid="stSidebar"] { background-color: #ffffff; box-shadow: 2px 0 5px rgba(0,0,0,0.05); }
    .stButton>button { border-radius: 25px; background-color: #ffffff; color: #007bff; border: 1px solid #007bff; padding: 0.4em 1.2em; font-weight: 600; transition: 0.3s; }
    .stButton>button:hover { background-color: #007bff; color: white; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


with st.sidebar:
    st.markdown("### ⚖️ Trợ lý Pháp luật AI")
    st.markdown("---")
    st.markdown(
        "Chào mừng bạn! Tôi là một trợ lý AI có khả năng tự động phân tích câu hỏi của bạn để lựa chọn phương pháp xử lý tốt nhất, bao gồm:"
    )
    st.markdown("""
    -   **Tra cứu nội bộ:** Tìm kiếm trong cơ sở dữ liệu văn bản pháp luật.
    -   **Tìm kiếm Web:** Truy cập Internet để có thông tin mới nhất và trích dẫn nguồn.
    -   **Phân rã & Viết lại:** Tự động đơn giản hóa các câu hỏi phức tạp.
    """)
    st.markdown("---")
    
    if st.button("🗑️ Bắt đầu cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

if not st.session_state.messages:
    st.markdown("""
        <div style='text-align: center; margin-top: 50px;'>
            <h2 style='color: #2c3e50;'>🤖 Tôi có thể giúp gì cho bạn?</h2>
            <p style='font-size: 18px; color: #555;'>Hãy nhập câu hỏi của bạn bên dưới, ví dụ:</p>
            <p><i>"Trách nhiệm của UBND cấp xã trong quản lý chợ?"</i> hoặc <i>"So sánh iPhone 15 và Samsung S24"</i></p>
        </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

if prompt := st.chat_input("💬 Nhập câu hỏi của bạn tại đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑‍💻"})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("⚖️ Trợ lý AI đang suy nghĩ và tìm kiếm..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                final_state = app.invoke({"messages": [HumanMessage(content=prompt)]}, config)
                response_text = final_state['messages'][-1].content
                
                st.markdown(response_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "avatar": "⚖️"
                })

            except Exception as e:
                error_message = f"❌ Đã xảy ra lỗi nghiêm trọng: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "avatar": "⚖️"})