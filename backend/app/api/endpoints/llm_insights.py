from fastapi import APIRouter, Depends, Body
from fastapi.concurrency import run_in_threadpool  # Import this utility
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services import database_service as db_service
from app.services import llm_service

router = APIRouter()

# Change the function signature to be 'async def'
@router.post("/ask")
async def ask_question(
    question: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """
    Receives a question, retrieves relevant traffic data,
    and uses an LLM to generate an answer asynchronously.
    """
    # 1. Retrieve data (this is fast, so no change needed)
    recent_logs = db_service.get_historical_logs(db)

    # 2. Run the slow, blocking LLM call in a background thread
    #    This prevents the server from freezing.
    #    The 'await' keyword tells our function to wait here until the
    #    background task is complete, without blocking other requests.
    answer = await run_in_threadpool(
        llm_service.generate_insights, question, recent_logs
    )

    return {"answer": answer}

