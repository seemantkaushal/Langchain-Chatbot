# my_chatbot/main.py

import os
import pickle
import faiss
import tempfile
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

def process_document(url=None, uploaded_file=None, user_question=None):
    try:
        docs = []
        faiss_index_file = "faiss_store.pkl"

        if os.path.exists(faiss_index_file):
            with open(faiss_index_file, "rb") as f:
                vectorStore = pickle.load(f)
                print("vectorstore already exists")
        else:
            if uploaded_file:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                # Load the documentx from the saved PDF file
                pdf_loader = PyPDFLoader(tmp_file_path)
                data = pdf_loader.load()
            elif url:
                # Load the document from the URL
                loaders = UnstructuredURLLoader(urls=[url])
                data = loaders.load()

            # Split documents into chunks
            text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(data)

            # Define the Hugging Face model for embeddings
            embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

            # Create FAISS vector store
            vectorStore = FAISS.from_documents(docs, embeddings)

            # Save the FAISS index
            with open(faiss_index_file, "wb") as f:
                pickle.dump(vectorStore, f)

        retriever = vectorStore.as_retriever()

        # Initialize the LLM for question-answering and summarization
        llm_model_name = "google/flan-t5-large"
        llm = HuggingFaceHub(repo_id=llm_model_name, model_kwargs={"temperature": 0.7,"max_length":500})

        # Create the RetrievalQAWithSourcesChain
        qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        if user_question:
            # Retrieve the answer to the user's question
            result = qa_chain({"question": user_question})
            answer = result['answer']
            sources = result['sources']
            return answer, sources
        else:
            return None, None

    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")
