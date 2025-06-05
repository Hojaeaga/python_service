"""
Centralized prompt management for the AI service
"""

# User Summary Workflow
USER_SUMMARY_PROMPT = """
Analyze the following user data and extract key information about their interests, expertise, and engagement patterns.
Focus on identifying topics they frequently engage with and their level of expertise in different areas.
Return a comma-separated list of relevant keywords and phrases that best describe the user's profile.

User Data:
{user_data}
"""

# Reply Generation Workflow
INTENT_CHECK_PROMPT = """
Analyze the following cast (social media post) and determine if it warrants a reply.
Consider factors like:
- Is it a question or request for help?
- Does it present an opportunity for meaningful engagement?
- Is there potential to add value through a response?

Cast Text:
{cast_text}

Return a JSON object with:
{
    "should_reply": true/false,
    "identified_needs": ["list", "of", "needs"],
    "confidence": 0.0-1.0
}
"""

CONTENT_DISCOVERY_PROMPT = """
Based on the cast and identified needs, find the most relevant content from the available feeds to include in a reply.
Consider:
- Relevance to the identified needs
- Recency and freshness of content
- Authority and credibility
- Potential impact and value-add

Cast Text:
{cast_text}

Identified Needs:
{identified_needs}

Available Feeds:
{feeds}

Return a JSON object with:
{
    "selected_content": {
        "title": "string",
        "url": "string",
        "relevance_score": 0.0-1.0,
        "key_points": ["list", "of", "points"]
    }
}
"""

REPLY_GENERATION_PROMPT = """
Generate a friendly and helpful reply to the cast using the selected content.
The reply should be:
- Concise and to the point
- Natural and conversational
- Helpful without being pushy
- Include relevant link(s) if available

Cast Text:
{cast_text}

Selected Content:
{selected_content}

Return a JSON object with:
{
    "reply_text": "string",
    "link": "string"
}
"""

# Embeddings Workflow
EMBEDDINGS_PROMPT = """
Prepare the following input data for embedding generation.
Extract and combine the most semantically meaningful elements while preserving the core meaning.
Clean and normalize the text, removing any noise or irrelevant information.

Input Data:
{input_data}
""" 