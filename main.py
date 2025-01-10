import os
from typing import List
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import PGVectorStore
from llama_index.storage.storage_context import StorageContext
import psycopg2
from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup

load_dotenv()

app = FastAPI()

def init_db():
    conn = psycopg2.connect(
        host="vector_store",
        database="llamaindex",
        user="admin",
        password="password"
    )
    vector_store = PGVectorStore.from_params(
        database=conn,
        table_name="embeddings",
        dimension=1536
    )
    return vector_store

def scrape_url(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

@app.post("/create_index")
async def create_index(urls: List[str]):
    texts = [scrape_url(url) for url in urls]
    
    vector_store = init_db()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents=[Document(text=text) for text in texts],
        storage_context=storage_context
    )
    return {"status": "success"}

@app.post("/query")
async def query_index(query: str):
    vector_store = init_db()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return {"response": str(response)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
