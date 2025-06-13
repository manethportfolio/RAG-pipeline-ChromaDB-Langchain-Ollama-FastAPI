# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from main import run_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    use_rag: bool = True

@app.post("/generate-response/")
def generate_response(req: QueryRequest):
    response = run_pipeline(
        step="step05_generate_response",
        query_args=req.question,
        use_rag=req.use_rag
    )
    return {"response": response}
