from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel

class WorkflowConfig(BaseModel):
    """Base configuration for workflows"""
    max_tokens: Optional[int] = None
    streaming: bool = False

class BaseWorkflow(ABC):
    """Base class for all workflows"""
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data and return results"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the workflow configuration"""
        pass 