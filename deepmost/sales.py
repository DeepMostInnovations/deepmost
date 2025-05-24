"""High-level API for sales conversion prediction"""

import os
import sys
from typing import List, Dict, Optional, Union
from .core.predictor import SalesPredictor
from .core.utils import download_model

# Model URLs based on Python version and backend
OPENSOURCE_MODEL_URL = "https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/resolve/main/sales_conversion_model.zip"
AZURE_MODEL_URL = "https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/resolve/main/sales_model_311.zip"

# Default paths
OPENSOURCE_MODEL_PATH = os.path.expanduser("~/.deepmost/models/sales_conversion_model.zip")
AZURE_MODEL_PATH = os.path.expanduser("~/.deepmost/models/sales_model.zip")


def _get_default_model_info(use_azure: bool = False):
    """Get model URL and path based on backend and Python version"""
    python_version = sys.version_info
    
    if use_azure:
        if python_version < (3, 10):
            raise RuntimeError("Azure OpenAI backend requires Python 3.10 or higher")
        return AZURE_MODEL_URL, AZURE_MODEL_PATH
    else:
        if python_version < (3, 11):
            raise RuntimeError("Open-source backend requires Python 3.11 or higher")
        return OPENSOURCE_MODEL_URL, OPENSOURCE_MODEL_PATH


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
        auto_download: bool = True,
        force_backend: Optional[str] = None
    ):
        """
        Initialize the sales agent.
        
        Args:
            model_path: Path to the PPO model. If None, downloads appropriate model.
            azure_api_key: Azure OpenAI API key (for Azure embeddings)
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure deployment name for embeddings
            embedding_model: HuggingFace model name for embeddings (ignored if using Azure)
            use_gpu: Whether to use GPU for inference
            llm_model: Optional LLM model path or HF repo for response generation
            auto_download: Whether to auto-download model if not found
            force_backend: Force 'azure' or 'opensource' backend (for testing)
        """
        # Determine backend
        if force_backend:
            self.use_azure = force_backend.lower() == 'azure'
        else:
            self.use_azure = all([azure_api_key, azure_endpoint, azure_deployment])
        
        # Handle model path
        if model_path is None:
            model_url, model_path = _get_default_model_info(self.use_azure)
            if not os.path.exists(model_path) and auto_download:
                print(f"Downloading {'Azure' if self.use_azure else 'open-source'} model to {model_path}...")
                download_model(model_url, model_path)
        
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
    
    def analyze_conversation_progression(
        self,
        conversation: Union[List[Dict[str, str]], List[str]],
        conversation_id: Optional[str] = None,
        print_results: bool = True
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        Analyze how conversion probability evolves turn by turn through a conversation.
        
        Args:
            conversation: List of messages (same format as predict method)
            conversation_id: Optional conversation ID for tracking
            print_results: Whether to print formatted results to console
        
        Returns:
            List of dicts with turn-by-turn analysis results
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
        
        results = []
        
        # Analyze each turn progressively
        for i in range(len(conversation)):
            # Get conversation up to current turn
            conversation_so_far = conversation[:i+1]
            
            # Get prediction for this turn
            result = self.predictor.predict_conversion(
                conversation_history=conversation_so_far,
                conversation_id=f"{conversation_id}_progression",  # Use unique ID for progression
                is_incremental_prediction=False  # Each turn is analyzed independently
            )
            
            current_msg = conversation[i]
            turn_result = {
                'turn': i + 1,
                'speaker': current_msg['speaker'],
                'message': current_msg['message'],
                'probability': result['probability'],
                'status': result['status'],
                'metrics': result['metrics']
            }
            
            results.append(turn_result)
            
            if print_results:
                # Format message for display (truncate if too long)
                display_msg = current_msg['message']
                if len(display_msg) > 60:
                    display_msg = display_msg[:57] + "..."
                
                print(f"Turn {i + 1} ({current_msg['speaker']}): \"{display_msg}\" -> Probability: {result['probability']:.4f}")
        
        if print_results:
            print(f"\nFinal Conversion Probability: {results[-1]['probability']:.2%}")
            print(f"Final Status: {results[-1]['status']}")
        
        return results
    
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


def analyze_progression(conversation: Union[List[Dict[str, str]], List[str]], **kwargs) -> List[Dict]:
    """
    Quick turn-by-turn analysis function.
    
    Example:
        from deepmost import sales
        results = sales.analyze_progression([
            "Hi, I need a CRM", 
            "Our CRM starts at $29/month",
            "That sounds interesting, tell me more"
        ])
    """
    agent = Agent(**kwargs)
    return agent.analyze_conversation_progression(conversation, print_results=True)


def get_system_info():
    """Get system information for debugging"""
    import sys
    import torch
    
    info = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'cuda_available': torch.cuda.is_available(),
        'supported_backends': []
    }
    
    # Check backend support
    try:
        _get_default_model_info(use_azure=False)
        info['supported_backends'].append('opensource')
    except RuntimeError:
        pass
    
    try:
        _get_default_model_info(use_azure=True)
        info['supported_backends'].append('azure')
    except RuntimeError:
        pass
    
    return info