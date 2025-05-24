"""High-level API for sales conversion prediction"""

import os
from typing import List, Dict, Optional, Union
from .core.predictor import SalesPredictor
from .core.utils import download_model

# Default model URL
DEFAULT_MODEL_URL = "https://github.com/DeepMostInnovations/sales-conversion-model-reinf-learning/raw/main/sales_conversion_model.zip"
DEFAULT_MODEL_PATH = os.path.expanduser("~/.deepmost/models/sales_conversion_model.zip")


class Agent:
    """Sales prediction agent with simple API"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        embedding_model: str = "BAAI/bge-m3",
        use_gpu: bool = True,
        llm_model: Optional[str] = None,
        auto_download: bool = True
    ):
        """
        Initialize the sales agent.
        
        Args:
            model_path: Path to the PPO model. If None, downloads default model.
            azure_api_key: Azure OpenAI API key (for Azure embeddings)
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure deployment name for embeddings
            embedding_model: HuggingFace model name for embeddings (ignored if using Azure)
            use_gpu: Whether to use GPU for inference
            llm_model: Optional LLM model path or HF repo for response generation
            auto_download: Whether to auto-download model if not found
        """
        # Handle model path
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
            if not os.path.exists(model_path) and auto_download:
                print(f"Downloading model to {model_path}...")
                download_model(DEFAULT_MODEL_URL, model_path)
        
        # Determine backend
        self.use_azure = all([azure_api_key, azure_endpoint, azure_deployment])
        
        # Initialize predictor
        self.predictor = SalesPredictor(
            model_path=model_path,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            embedding_model=embedding_model,
            use_gpu=use_gpu,
            llm_model=llm_model
        )
    
    def predict(
        self,
        conversation: Union[List[Dict[str, str]], List[str]],
        conversation_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Predict conversion probability for a conversation.
        
        Args:
            conversation: List of messages. Can be:
                - List of dicts with 'speaker' and 'message' keys
                - List of strings (alternating customer/sales_rep)
            conversation_id: Optional conversation ID for tracking
        
        Returns:
            Dict with 'probability' and other metrics
        """
        # Normalize conversation format
        if conversation and isinstance(conversation[0], str):
            # Convert list of strings to list of dicts
            normalized = []
            for i, msg in enumerate(conversation):
                speaker = "customer" if i % 2 == 0 else "sales_rep"
                normalized.append({"speaker": speaker, "message": msg})
            conversation = normalized
        
        # Generate conversation ID if not provided
        if conversation_id is None:
            import uuid
            conversation_id = str(uuid.uuid4())
        
        # Get prediction
        result = self.predictor.predict_conversion(
            conversation_history=conversation,
            conversation_id=conversation_id
        )
        
        return result
    
    def predict_with_response(
        self,
        conversation: Union[List[Dict[str, str]], List[str]],
        user_input: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Union[str, Dict]]:
        """
        Generate sales response and predict conversion probability.
        
        Args:
            conversation: Conversation history
            user_input: Latest user message
            conversation_id: Optional conversation ID
            system_prompt: Optional system prompt for LLM
        
        Returns:
            Dict with 'response' and 'prediction' keys
        """
        # Normalize conversation format
        if conversation and isinstance(conversation[0], str):
            normalized = []
            for i, msg in enumerate(conversation):
                speaker = "customer" if i % 2 == 0 else "sales_rep"
                normalized.append({"speaker": speaker, "message": msg})
            conversation = normalized
        
        if conversation_id is None:
            import uuid
            conversation_id = str(uuid.uuid4())
        
        return self.predictor.generate_response_and_predict(
            conversation_history=conversation,
            user_input=user_input,
            conversation_id=conversation_id,
            system_prompt=system_prompt
        )


# Convenience function for quick predictions
def predict(conversation: Union[List[Dict[str, str]], List[str]], **kwargs) -> float:
    """
    Quick prediction function.
    
    Example:
        from deepmost import sales
        probability = sales.predict(["Hi, I need a CRM", "Our CRM starts at $29/month"])
    """
    agent = Agent(**kwargs)
    result = agent.predict(conversation)
    return result['probability']