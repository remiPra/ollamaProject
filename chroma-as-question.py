from langchain_community import embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


model_local = ChatOllama(model="mistral")



from langchain_community.embeddings import HuggingFaceEmbeddings

import ollama


embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')
# embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./rag-huiles", embedding_function=embedding)
print(vector_db)
retriever = vector_db.as_retriever()
print(retriever)




# 4. After RAG
print("\n########\nAfter RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("je ne pense pas que l'on parle de migraine specifiquement dans le document mais trouve moi une association d huiles essentielles pour la migraine et surtout leur quantité et leur association"))

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Define the Ollama LLM function
# def ollama_llm(question, context):
#     formatted_prompt = f"Question: {question}\n\nContext: {context}"
#     print(formatted_prompt)
#     response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
#     return response['message']['content']

# # Define the RAG chain
# def rag_chain(question):
#     retrieved_docs = retriever.invoke(question)
#     formatted_context = format_docs(retrieved_docs)
#     return ollama_llm(question, formatted_context)

# # Use the RAG chain
# result = rag_chain("j'ai un ongle incarné comment faire avec les infos de ce document")
# print(result)