import pytest
import os
from api import app
from fastapi.testclient import TestClient
from App import get_pdf_text, get_text_chunks, get_vector_store, user_input
from dotenv import load_dotenv


load_dotenv()

client = TestClient(app)

def test_pdf_text_extraction():
    pdf_path = "pdfs/AI.pdf"
    with open(pdf_path, "rb") as pdf_file:
        pdf_text = get_pdf_text([pdf_file])
        assert len(pdf_text) > 0, "PDF text extraction failed: No text extracted."

def test_query_endpoint():
 
    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key is not None, "API key is not set. Please check your .env file."

   
    response = client.post("/query", json={"question": "What is AI?", "api_key": api_key})
    assert response.status_code == 200, f"Endpoint query failed: Status code is {response.status_code}. Error: {response.text}"
    assert "answer" in response.json(), "Endpoint response does not contain 'answer'."

def test_vector_store_creation():
    text_chunks = ["sample text 1", "sample text 2"]
    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key is not None, "API key is not set. Please check your .env file."

    get_vector_store(text_chunks, api_key)
    assert os.path.exists("faiss_index"), "Vector store creation failed: 'faiss_index' not found."
