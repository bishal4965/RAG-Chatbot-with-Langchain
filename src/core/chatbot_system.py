import os
import streamlit as st
from typing import List, Any, Optional

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate


from config.settings import settings
from ..tools.appointment_booking import AppointmentBookingTool
from ..tools.document_qa import DocumentQATool



class ChatbotSystem:
    """Main chatbot system class"""

    def __init__(self):
        self.llm = None
        self.qa_chain = None
        self.agent = None
        self.memory = None
        self.document_qa_tool = None
        self.appointment_tool = None

        self.setup_llm()
        self.setup_memory()
        self.setup_tools()

    def setup_llm(self) -> bool:
        """Initialize the LLM with Groq configuration"""
        try:
            self.llm = ChatGroq(
                temperature=0.7,
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.LLM_MODEL,
            )

            # Test connection
            test_response = self.llm.invoke([HumanMessage("Hello, this is a test.")])

            if test_response:
                print(test_response)
                print("LLM connection successful!")

            return True

        except Exception as e:
            st.error(f"Failed to initialize the LLM: {str(e)}")
            st.error("Please check your GROQ_API_KEY and internet connection.")

            return False

    
    def setup_memory(self) -> None:
        """Setup conversational memory"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    
    def setup_tools(self) -> None:
        """Initialize tools"""
        self.appointment_tool = AppointmentBookingTool()

    
    def setup_vector_store(self, documents: List[Any]) -> bool:
        """Setup vector store with documents"""
        try:
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)

            # Create a vector store
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=settings.CHROMA_DB_PATH
            )

            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are an assistant for question-answering tasks. " \
                    "Use the following pieces of retrieved context to answer the question." \
                    "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
                    ),
                HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
                ])


            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            self.qa_chain = (
                {
                    "context": vector_store.as_retriever() | format_docs,
                    "question": RunnablePassthrough(),
                }
                | chat_prompt
                | self.llm
                | StrOutputParser()
            )

            self.document_qa_tool = DocumentQATool(self.qa_chain)

            self.setup_agent()

            st.success("Documents loaded successfully! You can now ask questions about them.")

            return True

        except Exception as e:
            st.error(f"Error setting up vectorstore: {str(e)}")
            return False
        

    def setup_agent(self) -> bool:
        """Setup the agent with tools"""

        if not self.llm:
            return False
        
        tools = []

        if self.document_qa_tool:
            tools.append(self.document_qa_tool)

        if self.appointment_tool:
            tools.append(self.appointment_tool)
        

        # Create system prompt
        system_prompt = """You are a helpful AI assistant that can:
            1. Answer questions from uploaded documents
            2. Help users book appointments by collecting their information
            3. Have general conversations
            
            When a user asks to "call them", "book appointment", or mentions scheduling, use the appointment_booking tool.
            For questions about uploaded documents, use the document_qa tool.
            Always be polite and helpful."""
        
        # prompt_template = """
        #     You are a helpful AI assistant that can:
        #     1. Answer questions from uploaded documents
        #     2. Help users book appointments by collecting their information
        #     3. Have general conversations
            
        #     When a user asks to "call them", "book appointment", or mentions scheduling, use the appointment_booking tool.
        #     For questions about uploaded documents, use the document_qa tool.
        #     Always be polite and helpful.
            
        #     You have access to the following tools:
            
        #     {tools}
            
        #     Use the following format:
            
        #     Question: the input question you must answer
        #     Thought: you should always think about what to do
        #     Action: the action to take, should be one of [{tool_names}]
        #     Action Input: the input to the action
        #     Observation: the result of the action
        #     ... (this Thought/Action/Action Input/Observation can repeat N times)
        #     Thought: I now know the final answer
        #     Final Answer: the final answer to the original input question
            
        #     Begin!
            
        #     Question: {input}
        #     Thought: {agent_scratchpad}
        #     """

        prompt = PromptTemplate.from_template(system_prompt)
        
        if tools:
            try:
                self.agent = initialize_agent(
                    tools=tools,
                    llm=self.llm,
                    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                    memory=self.memory,
                    verbose=True,
                    prompt=prompt,
                    handle_parsing_errors=True
                )
                return True
            except Exception as e:
                st.error(f"Error setting up an agent: {str(e)}")
                return False
            
        return False
    

    def process_user_input(self, user_input: str) -> str:
        """Process user input and return response"""

        if not user_input or not user_input.strip():
            return "I didn't receive any input. How can I help you?"
        
        user_input = user_input.strip()

        try:
            # Check if this is a booking initiation request
            booking_keywords = ['book appointment', 'schedule', 'call me', 'contact me', 'appointment']
            is_booking_request = any(keyword in user_input.lower() for keyword in booking_keywords)

            if self.appointment_tool and (is_booking_request or self.appointment_tool.is_booking_active()):
                return self.appointment_tool._run(user_input)
            
            # Handle agent-based interactions for document QA and general queries
            elif self.agent:
                try:
                    response = self.agent.invoke(user_input)
                    return response
                except Exception as agent_error:
                    # Fallback to direct LLM if agent fails
                    st.warning(f"Agent encountered an issue: {str(agent_error)}\n Using direct LLM response...")
                    return self._direct_llm_response(user_input)
                
            else:
                return self._direct_llm_response(user_input)
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."


        

    def _direct_llm_response(self, user_input: str) -> str:
        """Get direct response from LLM without tools"""
        if not self.llm:
            return "I'm sorry, I am not properly initialized. Please refresh the page and try again."
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a helpful AI assistant. Be polite, professional, and concise."),
                HumanMessage(content=user_input)
            ])
            return response.content
        
        except Exception as e:
            return f"I'm having trouble processing your request: {str(e)}. Please try again."
        
    
    def reset_conversation(self) -> None:
        """Reset conversation memory and booking state"""
        if self.memory:
            self.memory.clear()
        if self.appointment_tool:
            self.appointment_tool.reset_booking()

        st.success('Conversation reset successfully!')


    def get_system_status(self) -> dict:
        """Get current system status for debugging"""
        
        return {
            'llm_initialized': self.llm is not None,
            'memory_initialized': self.memory is not None,
            'agent_initialized': self.agent is not None,
            'appointment_tool_available': self.appointment_tool is not None,
            'document_qa_available': self.document_qa_tool is not None,
            'booking_active': self.appointment_tool.is_booking_active() if self.appointment_tool else False,
            'booking_progress': self.appointment_tool.booking_progress() if self.appointment_tool else None
        }

    
