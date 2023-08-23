# A couple of magic strings
chatModel               = "llama2"
collection_name         = "second_brain"
collection_directory    = "./"

# Some general python niceities
from typing import List
from pydantic import BaseModel

# Support our embeddings, vectordatabase, and langchain needs
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

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

# Try to wrangle our LLM into answering as honestly as possible
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""

## Set up a simple chain for prompting
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Set up out LLM (Ollama) for streaming callback
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chatllm = Ollama(base_url="http://localhost:11434",
             model=chatModel,
             verbose=False,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Define out interface
# Read an article and store it
def read_web(url) : 
    data = WebBaseLoader(url).load()
    all_split = text_splitter.split_documents(data)
    ids = vectorstore.add_documents(all_split)
    return

# Learn a new fact
def learn(text) :
    document = Document(page_content=text, metadata={"source": "a little birdie"})
    vectorstore.add_documents([document])
    vectorstore.persist()
    return

class SourcedAnswer(BaseModel):
    answer: str
    sources: List[str]

# Query our second brain
def ask(question) :
    # Let's look at our sources, how reliable is the answer?
    # We do a similarity search, and discard documents with no source, or with too low of a score
    # NOTE:  THIS ONLY DISCORDS FROM CITATIONS, NOT FROM THE ANSWER
    docs = vectorstore.similarity_search_with_score(question)
    sources = [doc[0].metadata["source"] for doc in docs if hasattr(doc[0], "metadata") and doc[0].metadata and doc[1]<1]

    # Only give unique items.
    sources = list(set(sources))
    
    # Get the answer
    qa_chain = RetrievalQA.from_chain_type(
        chatllm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    answer =  qa_chain({"query": question})
    
    # Put it together nicely for our UI
    sourced_answer = SourcedAnswer(
        answer=answer["result"],
        sources=sources
    )
    return sourced_answer