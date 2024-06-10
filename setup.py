# setup.py
from setuptools import setup, find_packages

setup(
    name='my_chatbot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'faiss-cpu',
        'langchain',
        'sentence-transformers',
        'PyPDF2',  # If using PyPDFLoader
        # add other dependencies here
    ],
)
