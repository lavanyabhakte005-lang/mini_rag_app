from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_pipeline import run_rag


app = FastAPI(
    title="Mini RAG Application",
    description="Tourism PDF based RAG application",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Mini RAG API is running"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    answer = run_rag(request.question)

    return {
        "question": request.question,
        "answer": answer
    }