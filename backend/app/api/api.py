from fastapi import APIRouter
from app.api.endpoints import traffic_data, llm_insights # Assuming you have these files

api_router = APIRouter()

# This is correct. It includes the routes from the file we just modified.
api_router.include_router(traffic_data.router, prefix="/traffic", tags=["Traffic Data"])

# This line is also correct, assuming you have an llm_insights router.
api_router.include_router(llm_insights.router, prefix="/insights", tags=["LLM Insights"])