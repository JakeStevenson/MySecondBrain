# A couple of magic strings
collection_name         = "second_brain"
collection_directory    = "./"

# Some general python niceities
from pydantic import BaseModel
from datetime import datetime

# Support our embeddings, vectordatabase, and langchain needs
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

# Hide some useless warnings
import warnings
warnings.filterwarnings('ignore')

# Set up the vector database and client we will use
persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection(collection_name)

vectorstore = Chroma(client=persistent_client,
                     collection_name=collection_name,
                     embedding_function=GPT4AllEmbeddings(), 
                     persist_directory=collection_directory)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

# Define out interface
# Read an article and store it
def read_web(url) : 
    data = WebBaseLoader(url).load()
    all_split = text_splitter.split_documents(data)
    ids = vectorstore.add_documents(all_split)
    return

# Learn a new fact
def learn(text) :
    current_datetime = datetime.now()
    source = "Informed by user on " + current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    document = Document(page_content=text, metadata={"source": source})
    vectorstore.add_documents([document])
    vectorstore.persist()
    return

class SourcedAnswer(BaseModel):
    answer: str
    source: str
    score: float

# Query our second brain
def ask(question) :
    # Let's look at our sources, how reliable is the answer?
    # We do a similarity search, and discard documents with no source, or with too low of a score
    # NOTE:  THIS ONLY DISCARDS FROM CITATIONS, NOT FROM THE ANSWER
    dd = vectorstore.get()
    docs = vectorstore.similarity_search_with_score(question)
    sourcedanswers = []
    for doc in docs :
        sourcedanswers.append(SourcedAnswer(answer=doc[0].page_content, source=doc[0].metadata["source"], score=doc[1]))
    return sourcedanswers