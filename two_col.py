import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "fa61d3bcde7c8965f6b10c1b4995276bb452a0a75bdd96327046d2829b60bf5e"

def process_pdf(pdf_file):
    try:
        # Read PDF
        pdfreader = PdfReader(pdf_file)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        # Split text
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()

        # Create document search
        document_search = FAISS.from_texts(texts, embeddings)

        return document_search
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None



def main():
    st.set_page_config(layout="wide")

    st.title("Researchers Chatbot")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Hi")
        st.title("Ask anything related to the research papers")

        # Upload PDF files
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            queries = st.session_state.setdefault("queries", {})
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"Processing file: {uploaded_file.name}")
                document_search = process_pdf(uploaded_file)
                if document_search:
                    query_key = f"query_{i}"
                    queries[query_key] = st.text_input(f"Enter your question for file {i+1}:", value=queries.get(query_key, ""))
                    button_key = f"button_{i}"
                    if st.button(f"Get Answer for file {i+1}:", key=button_key):
                        try:
                            query = queries[query_key]
                            # Load question answering chain
                            chain = load_qa_chain(OpenAI(), chain_type="stuff")
                            # Perform question answering
                            docs = document_search.similarity_search(query)
                            answer = chain.run(input_documents=docs, question=query)
                            # Display answer
                            st.write("Answer:", answer)
                        except Exception as e:
                            st.error(f"Error processing question: {e}")

    
    with col2:
        st.write("Hi")

if __name__ == "__main__":
    main()
