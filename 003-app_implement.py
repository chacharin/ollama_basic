import streamlit as st
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Initialize the model and chains outside of the main function to avoid reloading them on each interaction
model = ChatOllama(model="llama3")
prompt = PromptTemplate.from_template(
    """
    <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context. 
    If you don't know the answer, then reply, No Context available for this question {input}. [/Instructions] </s> 
    [Instructions] Question: {input} 
    Context: {context} 
    Answer: [/Instructions]
    """
)

embedding = FastEmbedEmbeddings()
vector_store = Chroma(persist_directory="./knowledge", embedding_function=embedding)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.5,
    },
)

document_chain = create_stuff_documents_chain(model, prompt)
chain = create_retrieval_chain(retriever, document_chain)

def ask(query: str):
    result = chain.invoke({"input": query})
    return result["answer"], result.get("context", [])

# Streamlit app
st.title("แชทบอท ประวัติคอมพิวเตอร์")

user_input = st.text_input("กรอกคำถาม:", "")

if st.button('ส่งคำถาม'):
    answer, context = ask(user_input)
    st.write('คำตอบ:', answer)
    for doc in context:
        st.write("ที่มา:", doc.metadata["source"])

# To run this, navigate to the directory containing the script and run:
# streamlit run chatbot_app.py
