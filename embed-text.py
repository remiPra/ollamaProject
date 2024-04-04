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
from bing_search import rechercher_bing  # Importez la fonction de votre premier script

# model_local = ChatOllama(model="mistral")
model_local = ChatOllama(model="mistral")

query="psg"
urls = rechercher_bing(query)

# # 1. Split data into chunks
# urls = [
#     "https://www.fdesouche.com/","https://www.valeursactuelles.com/"
    
# ]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
)
retriever = vectorstore.as_retriever()


after_rag_template = """Answer the question in french based only on the following context:
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
print(after_rag_chain.invoke("with this context ,  write an article about the last match of Mbappe , i want full text of article becaue i send your work to my boss"))

# loader = PyPDFLoader("Ollama.pdf")
# doc_splits = loader.load_and_split()