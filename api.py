from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from App import get_conversational_chain, get_vector_store, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import logging


load_dotenv()


app = FastAPI()
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    question: str
    api_key: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    question = request.question
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="API key is missing in environment variables.")

    try:
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(question)

        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return {"answer": response["output_text"]}
    
    except Exception as e:
        logging.error("Error in query processing: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
