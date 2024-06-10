# my_chatbot/streamlit_app.py

import streamlit as st
import os
from my_chatbot import process_document

# Set up Hugging Face Hub API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initialize Streamlit app
st.title("Document-based QA and Summarization Chatbot")

# Input fields
url = st.text_input("Enter the URL of the document:")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
user_question = st.text_input("Ask a question about the document:")

if st.button("Submit"):
    if not (url or uploaded_file):
        st.error("Please enter a URL or upload a PDF file.")
    else:
        with st.spinner("Loading and processing the document..."):
            try:
                answer, sources = process_document(url, uploaded_file, user_question)

                if answer or sources:
                    st.subheader("Answer to your question")
                    st.write(answer)
                    st.write(sources)
                else:
                    st.info("Please enter a question to get an answer.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
