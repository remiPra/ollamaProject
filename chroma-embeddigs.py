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

# 1. Split data into chunks
# urls = [
#     "https://www.oraclesdivinatoires.com/",
    
# ]
# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
# doc_splits = text_splitter.split_documents(docs_list)


# 1 :exemple court
# loader = PyPDFLoader("Invoice.pdf")
# print(loader)
# doc_splits = loader.load_and_split()


# 2 : exemple long
loader = PyPDFLoader("huiles.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_splits = text_splitter.split_documents(docs)

# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    persist_directory="rag-huiles",
    
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
)

print(vectorstore._collection.count())


# retriever = vectorstore.as_retriever()



# # 4. After RAG
# print("\n########\nAfter RAG\n")
# after_rag_template = """Answer the question based only on the following context:
# {context}
# Question: {question}
# """
# after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
# after_rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | after_rag_prompt
#     | model_local
#     | StrOutputParser()
# )
# print(after_rag_chain.invoke("creer moi une huile essentielle qui peut soulager l'arthrose du genou "))
