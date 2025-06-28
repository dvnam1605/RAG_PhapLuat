import streamlit as st
import os
from dotenv import load_dotenv
from qabot import rag_with_query_transformation

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Hệ thống RAG Pháp Luật",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .result-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .context-container {
        max-height: 400px;
        overflow-y: auto;
        background-color: #f1f3f4;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.9em;
    }
    .query-transform-info {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
    }
    h1 {
        color: #0d47a1;
    }
    h3 {
        color: #1976d2;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("⚖️ Hệ thống RAG Pháp Luật")
    
    # Sidebar for app configuration
    with st.sidebar:
        st.header("Cấu hình hệ thống")
        
        # Choose query transformation type
        st.subheader("Chọn phương pháp biến đổi truy vấn")
        transformation_option = st.radio(
            "Phương pháp biến đổi:",
            ["Không biến đổi", "Viết lại truy vấn (Rewrite)", "Mở rộng truy vấn (Step Back)", "Phân tách truy vấn (Decompose)"],
            index=0
        )
        
        # Map radio button options to transformation types
        transformation_map = {
            "Không biến đổi": None,
            "Viết lại truy vấn (Rewrite)": "rewrite",
            "Mở rộng truy vấn (Step Back)": "step_back",
            "Phân tách truy vấn (Decompose)": "decompose"
        }
        
        transformation_type = transformation_map[transformation_option]
        
        # Information about each transformation method
        if transformation_type:
            st.info(
                {
                    "rewrite": "Viết lại truy vấn để làm rõ và cụ thể hóa nội dung pháp lý.",
                    "step_back": "Mở rộng truy vấn để bao quát các khía cạnh pháp lý liên quan.",
                    "decompose": "Phân tách truy vấn phức tạp thành các truy vấn đơn giản hơn."
                }[transformation_type]
            )
        
        st.divider()
        st.markdown("### Giới thiệu")
        st.markdown("""
        Hệ thống truy xuất thông tin pháp lý sử dụng công nghệ RAG (Retrieval-Augmented Generation) 
        giúp tìm kiếm và trả lời các câu hỏi dựa trên văn bản pháp luật Việt Nam.
        """)
    
    # Main content area
    st.header("Tra cứu thông tin pháp luật")
    
    # Query input
    query = st.text_area("Nhập câu hỏi pháp lý của bạn:", height=100)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Gửi", type="primary", use_container_width=True)
    
    # Process the query when the button is clicked
    if search_button and query:
        with st.spinner('Đang xử lý truy vấn...'):
            try:
                # Call the RAG function with the query and transformation type
                result = rag_with_query_transformation(query, transformation_type)
                
                # Display the result
                st.markdown("### Kết quả")
                
                # Output original and transformed queries if applicable
                if transformation_type:
                    with st.expander("Thông tin về biến đổi truy vấn", expanded=False):
                        st.markdown(f"**Truy vấn gốc:** {result['Câu hỏi gốc']}")
                        st.markdown(f"**Phương pháp biến đổi:** {transformation_option}")
                
                # Display the answer
                st.markdown("### Trả lời")
                st.markdown(f"{result['Trả lời']}")
                
                # Show the context in an expander
                with st.expander("Xem các văn bản pháp luật liên quan", expanded=False):
                    st.markdown("### Ngữ cảnh từ văn bản pháp luật")
                    st.markdown(f"<div class='context-container'>{result['Ngữ cảnh'].replace('\n', '<br>')}</div>", 
                                unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")
    elif search_button:
        st.warning("Vui lòng nhập câu hỏi để tìm kiếm.")
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Hệ thống RAG Pháp Luật © 2025</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()