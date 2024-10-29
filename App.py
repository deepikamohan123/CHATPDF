import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os


st.set_page_config(page_title="ChatPDF", layout="wide")


st.markdown("""
## ChatPDF: Quick Insights from Your Documents

ChatPDF is an interactive tool that helps you extract information from your PDF files. It uses advanced AI to read and understand your documents, allowing you to ask questions and get instant answers.

### How to Use ChatPDF

1. **Input Your API Key**: Get a Google API key to enable AI access. You can find it from https://makersuite.google.com/app/apikey.

2. **Upload PDFs**: Drag and drop your PDF files for analysis.

3. **Ask Questions**: After uploading, type your questions about the content, and receive quick, accurate responses.
""")


api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


def main():
    st.header("ChatPDF: Your Document Assistant")

    user_question = st.text_input("What would you like to know from your PDFs?", key="user_question")

    if user_question and api_key: 
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Options:")
        pdf_docs = st.file_uploader("Upload your PDF files here (multiple files allowed)", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Process PDFs", key="process_button") and api_key: 
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Processing complete!")



if __name__ == "__main__":
    main()