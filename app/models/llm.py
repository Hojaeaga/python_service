"""
Centralized LLM model configurations
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Base model instances - these should be used throughout the application
REASONING_MODEL = ChatOpenAI(
    model="o4-mini"  # Let it use the default temperature
)

GENERATION_MODEL = ChatOpenAI(
    model="gpt-4.1-mini"  # Let it use the default temperature
)

EMBEDDINGS_MODEL = OpenAIEmbeddings(
    model="text-embedding-3-small"
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