import os
import logging
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,Document
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SimpleNodeParser
from config import GROQ_API_KEY, LOG_FILE
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from legal_chunker import hybrid_legal_chunker

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5" )


#setup logging for easy debugging purposes and to record audit trail
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,format="%(asctime)s [%(levelname)s]: %(message)s")

#wrap the llm 
llm = Groq(api_key=GROQ_API_KEY,model="llama-3.1-8b-instant")

#initate memoery for easy memory management and storage 
memory = ChatMemoryBuffer.from_defaults(token_limit=800)
#declare global variables to store the vectordatabase and the query engine
index=None
chat_engine = None

#function to load the contract thats locally available on the users device and 
def load_contract(file_path):
    global index,chat_engine
    logging.info(f"Loadinf file from path{file_path}..")
    
    
    reader = SimpleDirectoryReader(input_files=[file_path])
    docs = reader.load_data()
    raw_text = "\n".join([doc.text for doc in docs])

    #chunking with hybrid chunker
    chunks = hybrid_legal_chunker(raw_text, max_chunk_size=800)

    #convert the chunks back into a document for easy processing
    from llama_index.core import Document
    docs = [Document(text=chunk) for chunk in chunks]

    #prompt template to limit usage for legal contract reading/querying only
    system_prompt = (
        "You are a legal contract assistant. "
        "Your job is to read the contract and answer questions in plain English. "
        "Always explain clauses, obligations, and risks clearly. "
        "Do NOT provide legal advice — only summarize or clarify what the text says."
    ) 
    index = VectorStoreIndex.from_documents(docs,embed_model = embed_model)
    chat_engine = index.as_chat_engine(llm=llm,chat_mode="condense_plus_context",memory = memory,system_prompt=system_prompt)
    if index is None:
        logging.info("contract upload: unsuccessful")
    logging.info("contract upload: successful")

    
#final function
def query_contract(question):
    global chat_engine
    if chat_engine is None:
        logging.info("query engine creation: unsuccesful")
    logging.info(f"user question: {question}")
    response = chat_engine.chat(question)
    logging.info(f"bot response: {response}")
    return str(response)







