"""
Node definitions for LangGraph workflows
"""

import json
from typing import Any, Dict, List

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .models.llm import get_reasoning_model, get_embeddings_model
from .prompts import (
    CONTENT_DISCOVERY_PROMPT,
    EMBEDDINGS_PROMPT,
    INTENT_CHECK_PROMPT,
    REPLY_GENERATION_PROMPT,
    USER_SUMMARY_PROMPT,
)

# Model configurations
REASONING_MODEL = ChatOpenAI(model="o4-mini")

GENERATION_MODEL = ChatOpenAI(model="gpt-4.1-mini")

EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")


# User Summary Nodes
async def process_user_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process raw user data and extract summary"""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(USER_SUMMARY_PROMPT),
            HumanMessagePromptTemplate.from_template("{user_data}"),
        ]
    )
    response = await get_reasoning_model().ainvoke(
        prompt.format_messages(user_data=json.dumps(state["user_data"], indent=2))
    )

    keywords = [kw.strip() for kw in response.content.lower().split(",")]
    state["user_summary"] = {"keywords": keywords, "raw_summary": response.content}
    return state


async def generate_user_embedding(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate embeddings from user summary"""
    summary_text = " ".join(state["user_summary"]["keywords"])
    embedding = await get_embeddings_model().aembed_query(summary_text)

    state["user_embedding"] = {"vector": embedding, "dimensions": len(embedding)}
    return state


# Reply Generation Nodes
async def check_reply_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if the cast warrants a reply"""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(INTENT_CHECK_PROMPT),
            HumanMessagePromptTemplate.from_template("{cast_text}"),
        ]
    )
    response = await get_reasoning_model().ainvoke(
        prompt.format_messages(cast_text=state["cast_text"])
    )

    try:
        intent_analysis = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback if response is not valid JSON
        intent_analysis = {
            "should_reply": False,
            "identified_needs": [],
            "confidence": 0.0,
        }

    state["intent_analysis"] = intent_analysis
    return state


async def discover_relevant_content(state: Dict[str, Any]) -> Dict[str, Any]:
    """Find relevant content from feeds"""
    if not state["intent_analysis"]["should_reply"]:
        state["discovered_content"] = None
        return state

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(CONTENT_DISCOVERY_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "Cast: {cast_text}\n" "Needs: {identified_needs}\n" "Feeds: {feeds}"
            ),
        ]
    )
    response = await REASONING_MODEL.ainvoke(
        prompt.format_messages(
            cast_text=state["cast_text"],
            identified_needs=json.dumps(state["intent_analysis"]["identified_needs"]),
            feeds=json.dumps(state["available_feeds"], indent=2),
        )
    )

    try:
        content_discovery = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback if response is not valid JSON
        content_discovery = {
            "selected_content": {
                "title": "",
                "url": "",
                "relevance_score": 0.0,
                "key_points": [],
            }
        }

    state["discovered_content"] = content_discovery
    return state


async def generate_reply(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the final reply"""
    if not state.get("discovered_content"):
        state["reply"] = {"reply_text": "No response needed for this cast.", "link": ""}
        return state

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(REPLY_GENERATION_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "Cast: {cast_text}\n" "Content: {selected_content}"
            ),
        ]
    )
    response = await GENERATION_MODEL.ainvoke(
        prompt.format_messages(
            cast_text=state["cast_text"],
            selected_content=json.dumps(
                state["discovered_content"]["selected_content"]
            ),
        )
    )

    try:
        reply_data = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback if response is not valid JSON
        reply_data = {
            "reply_text": "I apologize, but I couldn't generate a proper response at this time.",
            "link": "",
        }

    state["reply"] = reply_data
    return state


# Embeddings Generation Nodes
async def prepare_embedding_text(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare text for embedding generation"""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(EMBEDDINGS_PROMPT),
            HumanMessagePromptTemplate.from_template("{input_data}"),
        ]
    )
    response = await get_reasoning_model().ainvoke(
        prompt.format_messages(input_data=json.dumps(state["input_data"]))
    )

    state["prepared_text"] = response.content
    return state


async def generate_embedding(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate embeddings from prepared text"""
    embedding = await EMBEDDINGS_MODEL.aembed_query(state["prepared_text"])

    state["embedding"] = {"vector": embedding, "dimensions": len(embedding)}
    return state

