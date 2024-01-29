import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import *
from langchain.llms import HuggingFaceHub
from timescale_vector import client
from langchain.vectorstores.timescalevector import TimescaleVector
from datetime import timedelta

def get_conversation_chain(vectorstor):
    # chat_model = ChatOpenAI()
    chat_model = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(memory=memory,
                                                           llm=chat_model,
                                                           retriever = vectorstor.as_retriever())
    return conversation_chain



def get_pdf_text(pdf_docs):
    Text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            Text += page.extract_text()
    return Text

def get_text_chunks(text):
    text_spliter = RecursiveCharacterTextSplitter( 
        separators=['\n'],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        
    )
    chunk = text_spliter.split_text(text)
    return chunk
def create_embeddings(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings =HuggingFaceBgeEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def handle_query(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history=response['chat_history']
    for i ,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
    st.write(response)
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your chatbot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")
    user_question=st.text_input("Ask a question about your documents")
    if user_question:
        handle_query(user_question)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    st.write(bot_template.replace("{{MSG}}","Hey handsome!"), unsafe_allow_html=True)
    st.write(user_template.replace("{{MSG}}","Hey dude!"), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #pdf text
                raw_Text = get_pdf_text(pdf_docs)
                #text chunks
                text_chunks = get_text_chunks(raw_Text)
                # store vectors
                vectors = create_embeddings(text_chunks)
                # memory of conversation
                st.session_state.conversation =get_conversation_chain(vectors)



if __name__=='__main__':
    main()
