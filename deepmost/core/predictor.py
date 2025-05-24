"""Main predictor class that handles both Azure and open-source backends"""

import os
import logging
import numpy as np
import torch
from typing import List, Dict, Optional, Any
from stable_baselines3 import PPO
from .embeddings import EmbeddingProvider, OpenSourceEmbeddings, AzureEmbeddings
from .utils import CustomLN, ConversationState, normalize_conversation

logger = logging.getLogger(__name__)


class SalesPredictor:
    """Unified predictor for sales conversion"""
    
    def __init__(
        self,
        model_path: str,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: str = "2023-12-01-preview",
        embedding_model: str = "BAAI/bge-m3",
        use_gpu: bool = True,
        llm_model: Optional[str] = None,
        **kwargs
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load PPO model
        self.model = PPO.load(model_path, device=self.device)
        logger.info(f"Loaded model from {model_path}")
        
        # Determine embedding dimensions
        total_obs_dim = self.model.observation_space.shape[0]
        self.expected_embedding_dim = total_obs_dim - (5 + 1 + 10)  # metrics + turn + probs
        logger.info(f"Model expects embedding dimension: {self.expected_embedding_dim}")
        
        # Initialize embedding provider
        if azure_api_key and azure_endpoint and azure_deployment:
            logger.info("Using Azure OpenAI embeddings")
            self.embedding_provider = AzureEmbeddings(
                api_key=azure_api_key,
                endpoint=azure_endpoint,
                deployment=azure_deployment,
                api_version=azure_api_version,
                expected_dim=self.expected_embedding_dim
            )
        else:
            logger.info(f"Using open-source embeddings: {embedding_model}")
            self.embedding_provider = OpenSourceEmbeddings(
                model_name=embedding_model,
                device=self.device,
                expected_dim=self.expected_embedding_dim,
                llm_model=llm_model
            )
        
        # Conversation states
        self.conversation_states = {}
    
    def predict_conversion(
        self,
        conversation_history: List[Dict[str, str]],
        conversation_id: str
    ) -> Dict[str, Any]:
        """Predict conversion probability for a conversation"""
        
        # Normalize conversation
        normalized_history = normalize_conversation(conversation_history)
        
        # Get turn number
        turn_number = len(self.conversation_states.get(conversation_id, {}).get('probabilities', []))
        
        # Get embedding
        full_text = " ".join([msg['message'] for msg in normalized_history])
        embedding = self.embedding_provider.get_embedding(full_text, turn_number)
        
        # Get metrics
        metrics = self.embedding_provider.analyze_metrics(normalized_history, turn_number)
        
        # Get previous probabilities
        previous_probs = self.conversation_states.get(conversation_id, {}).get('probabilities', [])
        
        # Create state
        state = ConversationState(
            conversation_history=normalized_history,
            embedding=embedding,
            conversation_metrics=metrics,
            turn_number=turn_number,
            conversion_probabilities=previous_probs
        )
        
        # Predict
        observation = state.state_vector
        action, _ = self.model.predict(observation, deterministic=True)
        probability = float(np.clip(action[0], 0.0, 1.0))
        
        # Update state
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = {'probabilities': []}
        self.conversation_states[conversation_id]['probabilities'].append(probability)
        
        # Format result
        return {
            'probability': probability,
            'turn': turn_number,
            'metrics': metrics,
            'status': self._get_status(probability)
        }
    
    def generate_response_and_predict(
        self,
        conversation_history: List[Dict[str, str]],
        user_input: str,
        conversation_id: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response and predict conversion"""
        
        normalized_history = normalize_conversation(conversation_history)
        
        # Generate response if LLM available
        if hasattr(self.embedding_provider, 'generate_response'):
            response = self.embedding_provider.generate_response(
                normalized_history,
                user_input,
                system_prompt
            )
        else:
            response = "Thank you for your interest. How can I help you with your specific needs?"
        
        # Create new history with response
        new_history = normalized_history + [
            {'speaker': 'customer', 'message': user_input},
            {'speaker': 'sales_rep', 'message': response}
        ]
        
        # Predict
        prediction = self.predict_conversion(new_history, conversation_id)
        
        return {
            'response': response,
            'prediction': prediction
        }
    
    def _get_status(self, probability: float) -> str:
        """Get status indicator based on probability"""
        if probability >= 0.7:
            return "🟢 High"
        elif probability >= 0.5:
            return "🟡 Medium"
        elif probability >= 0.3:
            return "🟠 Low"
        else:
            return "🔴 Very Low"