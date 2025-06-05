"""
Reply Generation Workflow
"""
from typing import Dict, Any

from langgraph.graph import Graph

from ..nodes import check_reply_intent, discover_relevant_content, generate_reply

class ReplyGenerationWorkflow:
    """Workflow for generating contextual replies"""
    
    def __init__(self):
        self.graph = self._build_graph()
        
    def _build_graph(self) -> Graph:
        """Build the workflow graph"""
        # Create nodes
        nodes = {
            "check_intent": check_reply_intent,
            "discover_content": discover_relevant_content,
            "generate_reply": generate_reply
        }
        
        # Create graph
        graph = Graph()
        
        # Add nodes
        graph.add_node("check_intent", nodes["check_intent"])
        graph.add_node("discover_content", nodes["discover_content"])
        graph.add_node("generate_reply", nodes["generate_reply"])
        
        # Add edges
        graph.add_edge("check_intent", "discover_content")
        graph.add_edge("discover_content", "generate_reply")
        
        # Set entry and end points
        graph.set_entry_point("check_intent")
        graph.set_finish_point("generate_reply")
        
        # Compile
        return graph.compile()
    
    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow"""
        return await self.graph.ainvoke(inputs) 