from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.utils import Output, DocumentService, QdrantService

app = FastAPI()

"""
Please create an endpoint that accepts a query string, e.g., "what happens if I steal 
from the Sept?" and returns a JSON response serialized from the Pydantic Output class.
"""


class QueryBody(BaseModel):
    query: str = Field(..., description="The query string to search the document store")


@app.post("/query", response_model=Output)
async def query_endpoint(body: QueryBody) -> Output:
    '''
    This endpoint accepts a query string, e.g., "what happens if I steal from the Sept?",
    and returns a JSON response serialized from the Pydantic Output class.

    The document store is based on the laws.pdf file in the docs directory.
    '''
    service = DocumentService()
    docs = service.create_documents()
    index = QdrantService(k=3)
    index.connect()
    index.load(docs)
    result = index.query(body.query)
    return result