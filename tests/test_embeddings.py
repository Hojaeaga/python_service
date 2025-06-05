"""
Tests for the Embeddings workflow
"""
import unittest
from unittest.mock import patch, AsyncMock
import json

from app.workflows.embeddings import EmbeddingsWorkflow

class TestEmbeddingsWorkflow(unittest.TestCase):
    def setUp(self):
        self.workflow = EmbeddingsWorkflow()
        self.test_input = {
            "title": "Understanding AI",
            "content": "A comprehensive guide to artificial intelligence",
            "tags": ["AI", "machine learning", "guide"]
        }
        
    @patch('app.nodes.REASONING_MODEL.ainvoke')
    @patch('app.nodes.EMBEDDINGS_MODEL.aembed_query')
    async def test_embeddings_workflow(self, mock_embed, mock_reason):
        # Mock the AI responses
        mock_reason.return_value = AsyncMock(
            content="Understanding AI: A comprehensive guide to artificial intelligence. Topics: AI, machine learning"
        )
        mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]  # Mock embedding vector
        
        # Run the workflow
        result = await self.workflow.run({"input_data": self.test_input})
        
        # Verify the workflow processed the data correctly
        self.assertIn("prepared_text", result)
        self.assertIn("embedding", result)
        
        # Check embedding structure
        self.assertIn("vector", result["embedding"])
        self.assertIn("dimensions", result["embedding"])
        self.assertEqual(result["embedding"]["dimensions"], 4)
        
        # Verify the mocks were called
        mock_reason.assert_called_once()
        mock_embed.assert_called_once()
        
    @patch('app.nodes.REASONING_MODEL.ainvoke')
    @patch('app.nodes.EMBEDDINGS_MODEL.aembed_query')
    async def test_empty_input(self, mock_embed, mock_reason):
        # Mock the AI responses for empty data
        mock_reason.return_value = AsyncMock(content="")
        mock_embed.return_value = []
        
        empty_input = {
            "title": "",
            "content": "",
            "tags": []
        }
        
        # Run the workflow
        result = await self.workflow.run({"input_data": empty_input})
        
        # Verify the workflow handles empty data gracefully
        self.assertIn("prepared_text", result)
        self.assertEqual(result["prepared_text"], "")
        self.assertIn("embedding", result)
        self.assertEqual(result["embedding"]["dimensions"], 0)
        
    @patch('app.nodes.REASONING_MODEL.ainvoke')
    @patch('app.nodes.EMBEDDINGS_MODEL.aembed_query')
    async def test_malformed_input(self, mock_embed, mock_reason):
        # Test with missing fields
        malformed_input = {
            "title": "Just a title"
            # missing content and tags
        }
        
        mock_reason.return_value = AsyncMock(content="Just a title")
        mock_embed.return_value = [0.1]
        
        # Run the workflow
        result = await self.workflow.run({"input_data": malformed_input})
        
        # Verify the workflow handles malformed data gracefully
        self.assertIn("prepared_text", result)
        self.assertIn("embedding", result)

if __name__ == '__main__':
    unittest.main() 