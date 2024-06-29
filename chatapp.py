import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = "AIzaSyDOdmxN5a1r46nRZYykqN_u4D9pzfMMKRQ"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question, pdf_docs):
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  # Create a new vector store for each user query
  raw_text = get_pdf_text(pdf_docs)
  text_chunks = get_text_chunks(raw_text)
  new_db = get_vector_store(text_chunks)
  query_embedding = embeddings.get_embedding(user_question)
  scores, docs = new_db.search(query_embedding, k=10)

  chain = get_conversational_chain()

    
  response = chain(
      {"input_documents":docs, "question": user_question}
    , return_only_outputs=True)

  print(response)
  st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Awesome Bot", page_icon = ":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    with st.sidebar:

        st.image("img/Robot.jpg")
        st.write("---")
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): # user friendly message.
                st.success("Done")
        
        st.write("---")
        st.write("AI App created by @ ANKIT MISHRA")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded.. ✍️📝")

    if user_question and 'pdf_docs' in locals():
        user_input(user_question, pdf_docs)


    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © Copyright 2024 | Made with ❤️ by <a href="https://github.com/Awesome-koder/AwesomeBot" target="_blank">ANKIT MISHRA</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
main()
