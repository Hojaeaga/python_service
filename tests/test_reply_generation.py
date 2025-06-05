"""
Tests for the Reply Generation workflow
"""
import unittest
from unittest.mock import patch, AsyncMock
import json

from app.workflows.reply_generation import ReplyGenerationWorkflow

class TestReplyGenerationWorkflow(unittest.TestCase):
    def setUp(self):
        self.workflow = ReplyGenerationWorkflow()
        self.test_cast = "What's the best way to learn AI?"
        self.test_feeds = [
            {
                "text": "A comprehensive guide to learning AI",
                "url": "https://example.com/ai-guide",
                "author": "AI Expert",
                "timestamp": "2024-03-20T10:00:00Z"
            }
        ]
        
    @patch('app.nodes.REASONING_MODEL.ainvoke')
    @patch('app.nodes.GENERATION_MODEL.ainvoke')
    async def test_reply_generation_workflow_positive(self, mock_gen, mock_reason):
        # Mock the AI responses
        mock_reason.side_effect = [
            # Intent check response
            AsyncMock(content=json.dumps({
                "should_reply": True,
                "identified_needs": ["learning_resources", "guidance"],
                "confidence": 0.9
            })),
            # Content discovery response
            AsyncMock(content=json.dumps({
                "selected_content": {
                    "title": "AI Learning Guide",
                    "url": "https://example.com/ai-guide",
                    "relevance_score": 0.85,
                    "key_points": ["Start with basics", "Practice coding"]
                }
            }))
        ]
        
        mock_gen.return_value = AsyncMock(content=json.dumps({
            "reply_text": "Here's a great guide for learning AI!",
            "link": "https://example.com/ai-guide"
        }))
        
        # Run the workflow
        result = await self.workflow.run({
            "cast_text": self.test_cast,
            "available_feeds": self.test_feeds
        })
        
        # Verify the workflow processed the data correctly
        self.assertIn("intent_analysis", result)
        self.assertIn("discovered_content", result)
        self.assertIn("reply", result)
        
        # Check intent analysis
        self.assertTrue(result["intent_analysis"]["should_reply"])
        
        # Check discovered content
        self.assertIn("selected_content", result["discovered_content"])
        
        # Check reply
        self.assertIn("reply_text", result["reply"])
        self.assertIn("link", result["reply"])
        
        # Verify the mocks were called
        self.assertEqual(mock_reason.call_count, 2)
        mock_gen.assert_called_once()
        
    @patch('app.nodes.REASONING_MODEL.ainvoke')
    @patch('app.nodes.GENERATION_MODEL.ainvoke')
    async def test_reply_generation_workflow_negative(self, mock_gen, mock_reason):
        # Mock the AI responses for no-reply case
        mock_reason.return_value = AsyncMock(content=json.dumps({
            "should_reply": False,
            "identified_needs": [],
            "confidence": 0.2
        }))
        
        # Run the workflow
        result = await self.workflow.run({
            "cast_text": "Just a random thought",
            "available_feeds": []
        })
        
        # Verify the workflow handles no-reply case correctly
        self.assertIn("intent_analysis", result)
        self.assertFalse(result["intent_analysis"]["should_reply"])
        self.assertIsNone(result.get("discovered_content"))
        self.assertEqual(result["reply"]["reply_text"], "No response needed for this cast.")
        
        # Verify only intent check was called
        mock_reason.assert_called_once()
        mock_gen.assert_not_called()

if __name__ == '__main__':
    unittest.main() 