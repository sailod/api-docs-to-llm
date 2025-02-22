import os
from typing import List, Any
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
from llama_index.core import Document
from llama_index.readers.web import SimpleWebPageReader
from tree_sitter import Language, Parser
import psycopg2
from fastapi import FastAPI
import requests
from typing import Optional
from pydantic import BaseModel, Field

app = FastAPI()

class URLInput(BaseModel):
    urls: List[str]
    reprocess_urls: Optional[List[str]] = None

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
    
    # Create tables if they don't exist
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
    
    # Get URLs to process
    urls_to_process = input_data.urls
    
    # If reprocess_urls specified, remove those URLs from DB first
    if input_data.reprocess_urls:
        with conn.cursor() as cur:
            for url in input_data.reprocess_urls:
                cur.execute("DELETE FROM ingested_urls WHERE url = %s", (url,))
                cur.execute("DELETE FROM data_embeddings WHERE metadata_->>'doc_id' = %s", (url,))
        conn.commit()
        # Add reprocess_urls to urls_to_process if not already included
        urls_to_process.extend([url for url in input_data.reprocess_urls if url not in urls_to_process])
    
    # Filter out already ingested URLs that aren't marked for reprocessing
    with conn.cursor() as cur:
        cur.execute("SELECT url FROM ingested_urls")
        existing_urls = {row[0] for row in cur.fetchall()}
    
    new_urls = [url for url in urls_to_process if url not in existing_urls or (input_data.reprocess_urls and url in input_data.reprocess_urls)]
    
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
        "message": f"Indexed {len(new_urls)} documents. {len(input_data.urls) - len(new_urls)} were already indexed and not marked for reprocessing."
    }

@app.post("/create_codebase_index")
async def create_codebase_index(git_url: URLInput):
    # TODO: move this function work to a celery worker
    vector_store, conn = init_db("codebase_index")

    # Git clone requested repo
    
    # Load and parse Python, JS files
    # Load Tree-Sitter Language
    # For this line you need to first clone the relevant tree sitter parser repo
    # git clone https://github.com/tree-sitter/tree-sitter-python.git
    # git clone https://github.com/tree-sitter/tree-sitter-javascript.git
    # git clone https://github.com/tree-sitter/tree-sitter-typescript.git
    # git clone https://github.com/tree-sitter/tree-sitter-go.git

    Language.build_library(
        "build/my-languages.so",  # Output file
        [
            "tree-sitter-python",
            "tree-sitter-javascript",
            "tree-sitter-typescript",
            "tree-sitter-go"
        ]
    )
    # Load the compiled languages
    PYTHON_LANGUAGE = Language("build/my-languages.so", "python")
    JAVASCRIPT_LANGUAGE = Language("build/my-languages.so", "javascript")
    TYPESCRIPT_LANGUAGE = Language("build/my-languages.so", "typescript")
    GOLANG_LANGUAGE = Language("build/my-languages.so", "go")
    
    parser = Parser()
    parser.set_language(PYTHON_LANGUAGE)
    
    def extract_code_structure(code):
        tree = parser.parse(code.encode("utf-8"))
        return tree.root_node.sexp()
    
    # Apply to files
    documents = SimpleDirectoryReader("your_codebase_path").load_data()
    for doc in documents:
        doc.text += "\n\nParsed Code:\n" + extract_code_structure(doc.text)
    
    settings = init_services()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context
    )
        
    return {
        "status": "success", 
        "message": f"Indexed {len(new_urls)} documents."
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
