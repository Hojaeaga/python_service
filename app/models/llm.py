"""
Centralized LLM model configurations
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REASONING_MODEL = ChatOpenAI(model="o4-mini", api_key=OPENAI_API_KEY, temperature=1)

GENERATION_MODEL = ChatOpenAI(
    model="gpt-4.1-mini", api_key=OPENAI_API_KEY, temperature=1
)

EMBEDDINGS_MODEL = OpenAIEmbeddings(
    model="text-embedding-3-small", api_key=OPENAI_API_KEY
)


# Factory functions to ensure consistent model creation
def get_reasoning_model() -> ChatOpenAI:
    """Get the reasoning model instance"""
    return REASONING_MODEL


def get_generation_model() -> ChatOpenAI:
    """Get the generation model instance"""
    return GENERATION_MODEL


def get_embeddings_model() -> OpenAIEmbeddings:
    """Get the embeddings model instance"""
    return EMBEDDINGS_MODEL

