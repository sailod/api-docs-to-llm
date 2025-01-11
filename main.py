import os
from typing import List, Any
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
from llama_index.core import Document
from llama_index.readers.web import SimpleWebPageReader
import psycopg2
from fastapi import FastAPI
import requests
from typing import Optional
from pydantic import BaseModel, Field

app = FastAPI()

class URLInput(BaseModel):
    urls: List[str]

class QueryInput(BaseModel):
    query: str

class AkashLLM(CustomLLM):
    base_url: str = Field(description="Base URL for the LLM API")
    
    def __init__(self, **kwargs):
        base_url = os.getenv("LLM_BASE_URL")
        if not base_url:
            raise ValueError("LLM_BASE_URL environment variable must be set")
        super().__init__(base_url=base_url, **kwargs)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = requests.post(
            self.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.getenv("AKASH_API_KEY")
            },
            json={
                "messages": [{"role": "user", "content": prompt}],
                "model": "Meta-Llama-3-1-8B-Instruct-FP8"
            }
        )
        text = response.json()["choices"][0]["message"]["content"]
        return CompletionResponse(text=text)

    async def stream_complete(self, prompt: str, **kwargs) -> Any:
        # Implement streaming completion
        raise NotImplementedError("Streaming not implemented")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,  # Maximum context window size
            model_name="llama2",
            num_output=256,  # Maximum number of output tokens
        )

def init_services():
    llm = AkashLLM()
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    return Settings

def init_db():
    conn_str = "postgresql://admin:password@vector_store:5432/llamaindex"
    async_conn_str = "postgresql+asyncpg://admin:password@vector_store:5432/llamaindex"
    conn = psycopg2.connect(conn_str)
    
    # Create URLs table if not exists
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingested_urls (
                url TEXT PRIMARY KEY,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    
    vector_store = PGVectorStore(
        connection_string=conn_str,
        async_connection_string=async_conn_str,
        schema_name="public",
        table_name="embeddings",
        embed_dim=384
    )
    return vector_store, conn

@app.post("/create_index")
async def create_index(input_data: URLInput):
    vector_store, conn = init_db()
    
    # Filter out already ingested URLs
    with conn.cursor() as cur:
        cur.execute("SELECT url FROM ingested_urls")
        existing_urls = {row[0] for row in cur.fetchall()}
    
    new_urls = [url for url in input_data.urls if url not in existing_urls]
    
    if not new_urls:
        return {"status": "success", "message": "All URLs already indexed"}
    
    # Use SimpleWebPageReader instead of BeautifulSoup
    documents = SimpleWebPageReader().load_data(new_urls)
    
    settings = init_services()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context
    )
    
    # Record newly ingested URLs
    with conn.cursor() as cur:
        for url in new_urls:
            cur.execute("INSERT INTO ingested_urls (url) VALUES (%s)", (url,))
    conn.commit()
    
    return {
        "status": "success", 
        "message": f"Indexed {len(new_urls)} new documents. {len(input_data.urls) - len(new_urls)} were already indexed."
    }

@app.post("/query")
async def query_index(input_data: QueryInput):
    vector_store, conn = init_db()
    settings = init_services()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    
    query_engine = index.as_query_engine()
    response = query_engine.query(input_data.query)
    return {"response": str(response)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
