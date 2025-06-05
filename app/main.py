"""
Main FastAPI application
"""
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .workflows.user_summary import UserSummaryWorkflow
from .workflows.reply_generation import ReplyGenerationWorkflow
from .workflows.embeddings import EmbeddingsWorkflow

app = FastAPI(
    title="AI Reply Service",
    description="AI-powered reply recommendation service",
    version="0.1.0"
)

# Request/Response Models
class UserData(BaseModel):
    """User data for summary generation"""
    user_data: Dict

class UserSummaryResponse(BaseModel):
    """Response from user summary workflow"""
    keywords: List[str]
    raw_summary: str
    embedding: Dict[str, List[float]]

class ReplyRequest(BaseModel):
    """Request for reply generation"""
    cast_text: str
    available_feeds: Optional[List[Dict]] = []

class ReplyResponse(BaseModel):
    """Response from reply generation workflow"""
    should_reply: bool
    reply_text: Optional[str] = None
    link: Optional[str] = None
    confidence: float

class EmbeddingsRequest(BaseModel):
    """Request for embeddings generation"""
    input_data: Dict

class EmbeddingsResponse(BaseModel):
    """Response from embeddings workflow"""
    prepared_text: str
    embedding: Dict[str, List[float]]

# Workflow Instances
user_summary_workflow = UserSummaryWorkflow()
reply_workflow = ReplyGenerationWorkflow()
embeddings_workflow = EmbeddingsWorkflow()

@app.post("/user-summary", response_model=UserSummaryResponse)
async def generate_user_summary(request: UserData) -> Dict:
    """Generate user summary and embeddings"""
    try:
        result = await user_summary_workflow.run({"user_data": request.user_data})
        return {
            "keywords": result["user_summary"]["keywords"],
            "raw_summary": result["user_summary"]["raw_summary"],
            "embedding": {
                "vector": result["user_embedding"]["vector"],
                "dimensions": result["user_embedding"]["dimensions"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-reply", response_model=ReplyResponse)
async def generate_reply(request: ReplyRequest) -> Dict:
    """Generate reply for a cast"""
    try:
        result = await reply_workflow.run({
            "cast_text": request.cast_text,
            "available_feeds": request.available_feeds
        })
        
        return {
            "should_reply": result["intent_analysis"]["should_reply"],
            "reply_text": result["reply"]["reply_text"] if result["intent_analysis"]["should_reply"] else None,
            "link": result["reply"].get("link"),
            "confidence": result["intent_analysis"]["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-embeddings", response_model=EmbeddingsResponse)
async def generate_embeddings(request: EmbeddingsRequest) -> Dict:
    """Generate embeddings for input data"""
    try:
        result = await embeddings_workflow.run({"input_data": request.input_data})
        return {
            "prepared_text": result["prepared_text"],
            "embedding": {
                "vector": result["embedding"]["vector"],
                "dimensions": result["embedding"]["dimensions"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 