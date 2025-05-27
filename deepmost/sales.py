"""High-level API for sales conversion prediction"""

import os
import sys
import logging
from typing import List, Dict, Optional, Union
from .core.predictor import SalesPredictor
from .core.utils import download_model

# Set up logger
logger = logging.getLogger(__name__)

# Model URLs based on Python version and backend
OPENSOURCE_MODEL_URL = "https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/resolve/main/sales_conversion_model.zip"
AZURE_MODEL_URL = "https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning/resolve/main/sales_model_311.zip"

# Default paths
OPENSOURCE_MODEL_PATH = os.path.expanduser("~/.deepmost/models/sales_conversion_model.zip")
AZURE_MODEL_PATH = os.path.expanduser("~/.deepmost/models/sales_model.zip")


def _get_default_model_info(backend_type: str = "opensource"):
    """Get model URL and path based on backend type"""
    python_version = sys.version_info
    
    if backend_type == "azure":
        if python_version < (3, 10):
            raise RuntimeError("Azure OpenAI backend requires Python 3.10 or higher")
        return AZURE_MODEL_URL, AZURE_MODEL_PATH
    elif backend_type == "openai":
        if python_version < (3, 10):
            raise RuntimeError("OpenAI backend requires Python 3.10 or higher")
        return AZURE_MODEL_URL, AZURE_MODEL_PATH  # Same model for now
    else:  # opensource
        if python_version < (3, 11):
            raise RuntimeError("Open-source backend requires Python 3.11 or higher")
        return OPENSOURCE_MODEL_URL, OPENSOURCE_MODEL_PATH


class Agent:
    """Sales prediction agent with support for three backends: open-source, Azure OpenAI, and standard OpenAI"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        # Azure OpenAI parameters
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_chat_deployment: Optional[str] = None,
        azure_api_version: str = "2024-10-21",
        # Standard OpenAI parameters
        openai_api_key: Optional[str] = None,
        openai_embedding_model: str = "text-embedding-3-large",
        openai_chat_model: Optional[str] = None,
        # Open-source parameters
        embedding_model: str = "BAAI/bge-m3",
        llm_model: Optional[str] = None,
        use_gpu: bool = True,
        auto_download: bool = True,
        force_backend: Optional[str] = None
    ):
        """
        Initialize the sales agent with support for three backends.
        
        Backend Selection Priority:
        1. force_backend parameter
        2. Standard OpenAI (if openai_api_key provided)
        3. Azure OpenAI (if azure credentials provided)
        4. Open-source (default)
        
        Args:
            model_path: Path to the PPO model. If None, downloads appropriate model.
            
            # Azure OpenAI Backend
            azure_api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure deployment name for embeddings (e.g., "text-embedding-ada-002")
            azure_chat_deployment: Azure deployment name for chat completions (e.g., "gpt-4o")
            azure_api_version: Azure OpenAI API version (default: "2024-10-21")
            
            # Standard OpenAI Backend
            openai_api_key: Standard OpenAI API key
            openai_embedding_model: OpenAI embedding model (default: "text-embedding-3-large")
            openai_chat_model: OpenAI chat model (e.g., "gpt-4o", "gpt-4o-mini")
            
            # Open-source Backend
            embedding_model: HuggingFace model name for embeddings (default: "BAAI/bge-m3")
            llm_model: Optional LLM model path or HF repo for response generation
            use_gpu: Whether to use GPU for inference
            
            # General
            auto_download: Whether to auto-download model if not found
            force_backend: Force specific backend ('azure', 'openai', 'opensource')
        """
        # Determine backend
        if force_backend:
            self.backend_type = force_backend.lower()
            if self.backend_type not in ['azure', 'openai', 'opensource']:
                raise ValueError("force_backend must be 'azure', 'openai', or 'opensource'")
        elif openai_api_key:
            self.backend_type = 'openai'
        elif all([azure_api_key, azure_endpoint, azure_deployment]):
            self.backend_type = 'azure'
        else:
            self.backend_type = 'opensource'
        
        logger.info(f"Using {self.backend_type} backend")
        
        # Handle model path
        if model_path is None:
            model_url, model_path = _get_default_model_info(self.backend_type)
            if not os.path.exists(model_path) and auto_download:
                print(f"Downloading {self.backend_type} model to {model_path}...")
                download_model(model_url, model_path)
        elif model_path.startswith(('http://', 'https://')):
            # Handle URL: download to local cache
            import hashlib
            url_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
            local_model_path = os.path.expanduser(f"~/.deepmost/models/downloaded_{url_hash}.zip")
            
            if not os.path.exists(local_model_path) and auto_download:
                print(f"Downloading model from URL to {local_model_path}...")
                download_model(model_path, local_model_path)
            elif not os.path.exists(local_model_path):
                raise FileNotFoundError(f"Model URL provided but auto_download=False and local cache not found: {local_model_path}")
            
            model_path = local_model_path
        
        # Initialize predictor with appropriate backend
        if self.backend_type == 'azure':
            self.predictor = SalesPredictor(
                model_path=model_path,
                azure_api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                azure_chat_deployment=azure_chat_deployment,
                azure_api_version=azure_api_version,
                use_gpu=use_gpu
            )
        elif self.backend_type == 'openai':
            self.predictor = SalesPredictor(
                model_path=model_path,
                openai_api_key=openai_api_key,
                openai_embedding_model=openai_embedding_model,
                openai_chat_model=openai_chat_model,
                use_gpu=use_gpu
            )
        else:  # opensource
            self.predictor = SalesPredictor(
                model_path=model_path,
                embedding_model=embedding_model,
                llm_model=llm_model,
                use_gpu=use_gpu
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
                conversation_id=f"{conversation_id}_progression",
                is_incremental_prediction=False
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
            print(f"Backend: {self.backend_type.title()}")
        
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
        _get_default_model_info("opensource")
        info['supported_backends'].append('opensource')
    except RuntimeError:
        pass
    
    try:
        _get_default_model_info("azure")
        info['supported_backends'].append('azure')
    except RuntimeError:
        pass
        
    try:
        _get_default_model_info("openai")
        info['supported_backends'].append('openai')
    except RuntimeError:
        pass
    
    return info