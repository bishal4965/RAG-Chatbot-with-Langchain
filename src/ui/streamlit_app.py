import os
import streamlit as st
from typing import Any, List
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from ..core.chatbot_system import ChatbotSystem


class StreamlitUI:
    """Streamlit UI handler for the AI chatbot application"""

    def __init__(self):
        self.chatbot = None
        self.supported_file_types = ['pdf', 'txt']
        self.max_file_size_mb = 10

    def initialize_app(self) -> None:
        """Initialize the Streamlit application with configuration"""
        st.set_page_config(
            page_title="AI Chatbot with Document Q&A",
            page_icon="🤖",
            layout="wide",
        )

        # Simple custom CSS styling
        st.markdown("""
                    <style>
                    .main {
                        padding: 1rem;
                    }
                    .upload-section {
                        background: #f8f9fa;
                        padding: 1rem;
                        border-radius: 0.5rem;
                        margin-bottom: 1rem;
                    }
                    </style>
                    """, unsafe_allow_html=True)
        
        # Main title and description
        st.title("🤖 AI Chatbot Assistant")
        st.markdown("*Your intelligent assistant for document Q&A and appointment booking*")
        

    def render_document_upload(self) -> bool:
        """Render document upload section"""

        with st.container():
            st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
            st.subheader("📄 Upload Documents")

            uploaded_files = st.file_uploader(
                "Choose files to upload",
                accept_multiple_files=True,
                type=self.supported_file_types,
                help="Upload pdf or text files to ask questions about them"
            )

            if uploaded_files:
                st.info(f"📁 {len(uploaded_files)} file(s) selected")

                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("🔄 Process Documents", type="primary"):
                        return self._process_documents(uploaded_files)
                    
                with col2:
                    if st.button("Clear"):
                        return st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        return False
    

    def _process_documents(self, uploaded_files: List[Any]) -> bool:
        """Process uploaded documents"""

        if not self.chatbot:
            st.error("Chatbot system not available")
            return False
        
        with st.spinner("Processing documents..."):
            try:
                documents = []

                for uploaded_file in uploaded_files:
                    # Save file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        if uploaded_file.name.lower().endswith('pdf'):
                            loader = PyPDFLoader(temp_path)
                        else:
                            loader = TextLoader(temp_path, encoding='utf-8')

                        docs = loader.load()
                        documents.extend(docs)

                    finally:
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                # Setup vector store
                success = self.chatbot.setup_vector_store(documents)

                if success:
                    st.success(f"✅ Successfully processed {len(documents)} document chunks!")
                    return True
                else:
                    st.error("❌ Failed to process documents")
                    return False
                    
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                return False
            
    
    def render_chat_interface(self) -> None:
            """Render the main chat interface"""
            # Initialize messages if not exists
            if 'messages' not in st.session_state:
                st.session_state.messages = []

            # Display welcome message if no chat history
            if not st.session_state.messages:
                with st.chat_message("assistant"):
                    st.markdown("""
                    👋 **Hi! I'm your AI assistant.**

                    I can help you with:<br>

                    • **Document Q&A** - Ask questions about uploaded documents <br>
                    • **Appointment Booking** - Say "book appointment" to schedule <br>
                    • **General Questions** - Ask me anything!

                    What can I help you with today?
                    """, unsafe_allow_html=True)

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Type your message here..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = self.chatbot.process_user_input(prompt)
                        except Exception as e:
                            response = f"I encountered an error: {str(e)}. Please try again."

                    # cleaned_response = response['output'] if isinstance(response, dict) and 'output' in response else str(response)
                    
                    if isinstance(response, dict):
                        if 'output' in response:
                            cleaned_response = response['output']
                        elif 'result' in response:
                            cleaned_response = response['result']
                        else:
                            cleaned_response = str(response)
                    elif response == "Agent stopped due to iteration limit or time limit.":
                        cleaned_response = "I apologize, but I'm having trouble finding a comprehensive answer to your question in the uploaded documents. The information might not be available in the current documents, or it might be phrased differently. Could you try rephrasing your question or asking about a more specific aspect?"
                    else:
                        cleaned_response = str(response)
                    st.markdown(cleaned_response)

                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})


    def render_sidebar(self) -> None:
            """Render sidebar with controls"""
            with st.sidebar:
                st.header("Controls")

                # System status
                if self.chatbot:
                    status = self.chatbot.get_system_status()

                    st.subheader("Status")
                    if status.get('llm_initialized'):
                        st.success("✅ System Ready")
                    else:
                        st.error("❌ System Error")

                    if status.get('document_qa_available'):
                        st.info("📚 Documents Loaded")

                    if status.get('booking_active'):
                        progress = status.get('booking_progress', {})
                        st.warning(f"📞 Booking Active: {progress.get('step', 'unknown')}")

                st.divider()

                # Control buttons
                if st.button("🔄 Reset Chat", use_container_width=True):
                    if self.chatbot:
                        self.chatbot.reset_conversation()
                    st.session_state.messages = []
                    st.success("Chat reset!")
                    st.rerun()


                st.divider()

                # Quick help
                with st.expander("💡 Quick Help"):
                    st.markdown("""
                    **Document Q&A:**
                    - Upload PDF/text files above
                    - Ask questions about the content

                    **Appointment Booking:**
                    - Say "book appointment" or "call me"
                    - Follow the prompts for name, phone, email, date
                    - Use phrases like "next Monday" or "tomorrow"
   
                    """)


    def initialize_session_state(self) -> None:
            """Initialize session state"""
            if 'chatbot' not in st.session_state:
                with st.spinner("Initializing AI system..."):
                    try:
                        st.session_state.chatbot = ChatbotSystem()
                        if not st.session_state.chatbot.llm:
                            st.error("Failed to initialize AI system. Please check your configuration.")
                            st.stop()
                    except Exception as e:
                        st.error(f"Initialization error: {str(e)}")
                        st.stop()

            self.chatbot = st.session_state.chatbot


    def run(self) -> None:
            """Main application entry point"""

            self.initialize_app()

            self.initialize_session_state()

            self.render_sidebar()

            self.render_document_upload()

            self.render_chat_interface()

    

def run_app():
    """Run the Streamlit application"""
    app = StreamlitUI()
    app.run()

