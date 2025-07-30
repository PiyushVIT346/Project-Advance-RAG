import os
import streamlit as st
import tempfile
import numpy as np

from components.rag_pipeline import AdaptiveRAGSystem

# ---- Streamlit App ----

def main():
    st.set_page_config(
        page_title="Advanced Adaptive RAG System",
        page_icon="🧠",
        layout="wide"
    )
    st.title("🧠 Advanced Adaptive RAG System")
    st.markdown("""
    Upload a PDF or Word document and ask questions. The system automatically selects 
    the best retrieval and reranking strategy based on your query type.
    """)

    with st.sidebar:
        st.header("🔧 Configuration")
        gemini_api_key = st.text_input(
            "Gemini API Key", type="password",
            help="Required for document analysis and answer generation"
        )
        cohere_api_key = st.text_input(
            "Cohere API Key (Optional)", type="password",
            help="Optional: Enables advanced reranking with Cohere"
        )
        if not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API key to continue.")
            return

    if 'rag_system' not in st.session_state:
        from logging_config.logger import logger
        try:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = AdaptiveRAGSystem(gemini_api_key, cohere_api_key)
            st.success("✅ RAG system initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            return

    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=['pdf', 'docx', 'doc'],
        help="Upload a PDF or Word document to query against"
    )

    if uploaded_file:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            try:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                st.session_state.rag_system.load_documents(tmp_file_path, file_ext)
                num_chunks = len(st.session_state.rag_system.documents)
                st.success(f"✅ Successfully processed document into {num_chunks} chunks!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document Chunks", num_chunks)
                with col2:
                    avg_length = np.mean([len(doc.page_content) for doc in st.session_state.rag_system.documents])
                    st.metric("Avg Chunk Length", f"{avg_length:.0f} chars")
                with col3:
                    total_length = sum(len(doc.page_content) for doc in st.session_state.rag_system.documents)
                    st.metric("Total Content", f"{total_length:,} chars")
            except Exception as e:
                st.error(f"❌ Error processing document: {e}")
            finally:
                os.unlink(tmp_file_path)

    if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system.documents:
        st.header("❓ Ask Questions")
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Ask any question about your uploaded document..."
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Get Answer", type="primary")
        with col2:
            if st.button("🧹 Clear Results"):
                if 'query_results' in st.session_state:
                    del st.session_state.query_results

        if ask_button and query.strip():
            with st.spinner("Processing your question..."):
                result = st.session_state.rag_system.query(query)
                st.session_state.query_results = result

        if 'query_results' in st.session_state:
            result = st.session_state.query_results
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.subheader("📝 Answer")
                st.write(result["answer"])
                st.subheader("📊 Processing Details")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Query Type", result["query_type"].replace('_', ' ').title())
                with col2:
                    st.metric("Retrieval Strategy", result["retrieval_strategy"].replace('_', ' ').title())
                with col3:
                    st.metric("Reranking Method", result["reranking_method"].replace('_', ' ').title())
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Documents Retrieved", result["num_retrieved"])
                with col5:
                    st.metric("Final Documents Used", result["num_final"])
                if st.checkbox("📄 Show Retrieved Documents", key="show_docs"):
                    st.subheader("Retrieved Document Chunks")
                    for i, doc in enumerate(result["retrieved_documents"]):
                        with st.expander(f"📄 Document Chunk {i+1}", expanded=False):
                            st.write(doc["content"])
                            if doc["metadata"]:
                                st.subheader("Metadata")
                                for key, value in doc["metadata"].items():
                                    st.write(f"**{key}:** {value}")

        elif ask_button and not query.strip():
            st.warning("⚠️ Please enter a question to get started.")

    else:
        st.info("👆 Please upload a document to start asking questions!")

    with st.expander("ℹ️ About This System"):
        st.markdown("""
        (Place your about-markdown here, see previous answer for inspiration.)
        """)

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, LangChain, and Google Gemini")

if __name__ == "__main__":
    main()
