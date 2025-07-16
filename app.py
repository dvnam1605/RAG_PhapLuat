import streamlit as st
from dotenv import load_dotenv


from qabot import rag_pipeline 

load_dotenv()

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Trợ lý Pháp luậte",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(to bottom right, #f2f6fc, #e8ecf4);
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    .stButton>button {
        border-radius: 25px; background-color: #ffffff; color: #007bff;
        border: 1px solid #007bff; padding: 0.4em 1.2em; font-weight: 600; transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #007bff; color: white;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
   
    st.markdown("### Trợ lý Pháp luật AI")
    st.markdown("---")
    
    st.header("⚙️ Tùy chọn truy xuất")

    # Cập nhật các tùy chọn
    transformation_map = {
        "Không biến đổi (Regular)": None,
        "Viết lại thành câu khái quát (Rewrite)": "rewrite",
        "Phân rã câu hỏi phức tạp (Decompose)": "decompose",
    }

    transformation_option = st.radio(
        label="Phương pháp biến đổi (áp dụng cho tra cứu pháp luật):",
        options=transformation_map.keys(),
        index=0,
        help="Lựa chọn cách hệ thống xử lý câu hỏi của bạn trước khi tìm kiếm."
    )

    st.session_state.transformation_type = transformation_map[transformation_option]

    st.markdown("---")
    
    if st.button("🗑️ Xóa cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.info(
        "Hệ thống tự động định tuyến câu hỏi để tra cứu "
        "nội bộ hoặc tìm kiếm trên Internet."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.messages:
    st.markdown("""
        <div style='text-align: center; margin-top: 50px;'>
            <h2 style='color: #2c3e50;'>🤖 Chào mừng bạn đến với Trợ lý Pháp luật AI</h2>
            <p style='font-size: 18px; color: #555;'>Hãy nhập câu hỏi của bạn bên dưới, ví dụ:</p>
            <p><i>"Trách nhiệm của UBND cấp xã trong quản lý chợ?"</i> hoặc <i>"Giá xăng hôm nay?"</i></p>
        </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])
        if "details" in message:
            with st.expander("🔍 Xem chi tiết quá trình xử lý", expanded=False):
                st.info(f"**Phương pháp được chọn tự động:** {message['details']}")

if prompt := st.chat_input("💬 Nhập câu hỏi của bạn tại đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑‍💻"})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("⚖️ Trợ lý AI đang phân tích và tìm kiếm..."):
            try:
                result = rag_pipeline(
                    prompt, 
                    st.session_state.history, 
                    st.session_state.transformation_type
                )
                
                response_text = result.get("Trả lời", "Xin lỗi, đã có lỗi xảy ra.")
                method_used = result.get('Phương pháp', 'Không rõ')

                st.markdown(response_text)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "avatar": "⚖️",
                    "details": method_used
                })
                
                st.session_state.history.append((prompt, response_text))

            except Exception as e:
                error_message = f"❌ Đã xảy ra lỗi nghiêm trọng: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "avatar": "⚖️"})