import os
import time as t
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
import google.generativeai as genai  # Import the genai module

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("Google API key not found in environment variables.")

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def process_documents(documents_folder):
    documents = []
    
    for file_name in os.listdir(documents_folder):
        file_path = os.path.join(documents_folder, file_name)
        
        # Process each file type
        if file_name.endswith(".pdf"):
            documents += [page.extract_text() for page in PdfReader(file_path).pages]
        elif file_name.endswith(".txt"):
            with open(file_path, "r") as file:
                documents.append(file.read())
        elif file_name.endswith(".docx"):
            documents += [paragraph.text for paragraph in Document(file_path).paragraphs]
        elif file_name.endswith(".xlsx"):
            for sheet in pd.ExcelFile(file_path).sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                documents += df.dropna().astype(str).values.flatten().tolist()
        elif file_name.endswith(".pptx"):
            for slide in Presentation(file_path).slides:
                documents += [shape.text for shape in slide.shapes if hasattr(shape, "text")]
    
    # Split documents into chunks
    text = "\n".join(documents)
    if not text:
        return None
    
    text_chunks = CharacterTextSplitter(separator="\n", chunk_size=10000, chunk_overlap=500, length_function=len).split_text(text)
    if not text_chunks:
        return None

    # Create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(text_chunks, embeddings)

def answer_query(query, docsearch, chain):
    docs = docsearch.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

MAX_FILE_SIZE_MB = 2

def main():
    st.set_page_config(page_title="Document Chat Bot")
    
    # Set the title with Londrina Sketch font
    st.markdown("<h1 style='font-family: \"Londrina Sketch\";'>DOCUMENT SEARCH BOT</h1>", unsafe_allow_html=True)

    session_state = SessionState(documents_folder=None, docsearch=None, response="", is_admin=False)

    # Move User authentication to main window
    user_type = st.radio("Select user type:", ["User", "Admin"])
    if user_type == "Admin":
        admin_password = st.text_input("Enter admin password:", type="password")
        if admin_password == "admin@123":
            session_state.is_admin = True
            st.success("Authentication successful")
        elif admin_password:
            st.error("Incorrect password. Please try again.")

    documents_folder = "admin_uploaded_docs"
    
    # Move Admin upload functionality to the sidebar
    if session_state.is_admin:
        st.sidebar.header("Upload Documents")
        uploaded_files = st.file_uploader("Upload your documents📁", accept_multiple_files=True, type=["pdf", "txt", "docx", "xlsx", "pptx"])
        
        if not os.path.exists(documents_folder):
            os.makedirs(documents_folder)
        
        for file in uploaded_files:
            if len(file.getvalue()) > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File size exceeds {MAX_FILE_SIZE_MB} MB limit: {file.name}")
                continue
            with open(os.path.join(documents_folder, file.name), "wb") as f:
                f.write(file.getvalue())

        # Load QA chain
        prompt = PromptTemplate(template=""" 
            Answer the question as detailed as possible from the provided context, 
            if the answer is not in provided context say, "answer is not available in the context". 

            Context:\n {context}?\n
            Question:\n{question}\n
            Answer:
        """, input_variables=["context", "question"])
        
        session_state.docsearch = process_documents(documents_folder)
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        # Chat interface
        uploaded_documents = os.listdir(documents_folder)
        if uploaded_documents:
            st.subheader("AI Bot")
            user_input = st.text_input("You:", key="user_input")
            if st.button("Send"):
                response = answer_query(user_input, session_state.docsearch, chain)
                st.text("Bot:")
                with st.spinner("Wait for a moment!"):
                    t.sleep(2)
                    st.write(response)
            
            # Document management
            st.sidebar.header("Uploaded Documents:")
            selected_documents = [file_name for file_name in uploaded_documents if st.sidebar.checkbox(file_name)]
            if st.sidebar.button("Delete Selected Documents"):
                for file_name in selected_documents:
                    os.remove(os.path.join(documents_folder, file_name))
                st.sidebar.success("Documents deleted successfully.")
        else:
            st.warning("No documents uploaded yet.")

    else:
        # Normal user functionality
        if not os.path.exists(documents_folder) or not os.listdir(documents_folder):
            st.warning("No documents uploaded yet! Please wait for the admin to upload documents.")
            return

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=""" 
            Answer the question as detailed as possible from the provided context, 
            if the answer is not in provided context say, "answer is not available in the context". 

            Context:\n {context}?\n
            Question:\n{question}\n
            Answer:
        """, input_variables=["context", "question"])
        
        session_state.docsearch = process_documents(documents_folder)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        st.sidebar.header("Available Documents for Search:")
        for file_name in os.listdir(documents_folder):
            st.sidebar.write(file_name)

        st.subheader("AI Bot")
        user_input = st.text_input("You:", key="user_input")
        if st.button("Send"):
            response = answer_query(user_input, session_state.docsearch, chain)
            st.text("Bot:")
            with st.spinner("Wait for a moment!"):
                t.sleep(2)
                st.write(response)

    if st.sidebar.button("Refresh Page"):
        st.experimental_rerun()

if __name__ == '__main__':
    main()
