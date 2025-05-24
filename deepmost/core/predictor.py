"""Main predictor class that handles both Azure and open-source backends"""

import os
import logging
import numpy as np
import torch
from typing import List, Dict, Optional, Any, Union, Tuple # CORRECTED: Added Tuple
from stable_baselines3 import PPO
from .embeddings import EmbeddingProvider, OpenSourceEmbeddings, AzureEmbeddings
from .utils import ConversationState 

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
    ):
        self.ppo_device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.ppo_device} for PPO model inference.")
        self.inference_device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.inference_device} for potential embedding/LLM operations.")

        if not os.path.exists(model_path):
            logger.error(f"PPO Model path does not exist: {model_path}")
            raise FileNotFoundError(f"PPO Model not found at {model_path}")

        logger.info(f"Loading PPO model from {model_path}")
        try:
            self.model = PPO.load(model_path, device=self.ppo_device)
            logger.info(f"PPO Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load PPO model from {model_path}: {e}")
            raise

        if not hasattr(self.model, 'observation_space') or self.model.observation_space is None:
            logger.error("PPO Model does not have an observation_space.")
            raise ValueError("Loaded PPO model is invalid (missing observation_space).")

        total_obs_dim = self.model.observation_space.shape[0]
        num_metrics = 5
        num_turn_info = 1
        num_prev_probs = 10
        self.expected_embedding_dim = total_obs_dim - (num_metrics + num_turn_info + num_prev_probs)
        
        if self.expected_embedding_dim <= 0:
            logger.error(
                f"Calculated non-positive expected_embedding_dim ({self.expected_embedding_dim}) "
                f"from total_obs_dim ({total_obs_dim}). Check PPO model structure."
            )
            raise ValueError("Invalid PPO model observation space structure.")
        logger.info(f"PPO Model expects total_obs_dim: {total_obs_dim}, calculated expected_embedding_dim: {self.expected_embedding_dim}")

        self.use_azure_embeddings = bool(azure_api_key and azure_endpoint and azure_deployment)

        if self.use_azure_embeddings:
            logger.info("Using Azure OpenAI embeddings.")
            try:
                self.embedding_provider: EmbeddingProvider = AzureEmbeddings(
                    api_key=azure_api_key,
                    endpoint=azure_endpoint,
                    deployment=azure_deployment,
                    api_version=azure_api_version,
                    expected_dim=self.expected_embedding_dim
                )
            except Exception as e:
                logger.error(f"Failed to initialize AzureEmbeddings: {e}")
                raise
        else:
            logger.info(f"Using open-source embeddings with model: {embedding_model}.")
            if llm_model:
                logger.info(f"Open-source LLM for metrics/response: {llm_model}")
            else:
                logger.warning(
                    "No LLM model for open-source backend. Metrics use defaults, responses basic. "
                    "Accuracy may be affected if PPO model trained with LLM-derived metrics."
                )
            try:
                self.embedding_provider: EmbeddingProvider = OpenSourceEmbeddings(
                    model_name=embedding_model,
                    device=self.inference_device,
                    expected_dim=self.expected_embedding_dim,
                    llm_model=llm_model
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenSourceEmbeddings: {e}")
                raise

        self.conversation_states: Dict[str, Dict[str, Any]] = {}
        logger.info("SalesPredictor initialized successfully.")

    def _get_effective_turn_for_prediction(
        self,
        conversation_history: List[Dict[str,str]],
        conversation_id: str,
        is_incremental_call: bool 
        ) -> Tuple[int, List[float]]: # Type hint uses Tuple
        """
        Determines the effective turn number and previous probabilities for the current prediction.
        - For incremental calls (part of an ongoing conversation): uses stored state.
        - For one-shot full history calls: calculates turn from history length, previous_probs is empty.
        """
        if is_incremental_call:
            stored_state = self.conversation_states.get(
                conversation_id,
                {'probabilities': [], 'turn_number': 0} 
            )
            effective_turn = stored_state['turn_number']
            previous_probs = stored_state['probabilities']
            logger.debug(f"Incremental call for conv_id '{conversation_id}'. Effective turn from state: {effective_turn}. Prev_probs count: {len(previous_probs)}")
        else:
            if conversation_history:
                effective_turn = len(conversation_history) - 1
                if effective_turn < 0 : effective_turn = 0 
            else:
                effective_turn = 0 
            previous_probs = [] 
            logger.debug(f"One-shot/new call for conv_id '{conversation_id}'. Effective turn from history len: {effective_turn}. Prev_probs empty.")
        
        return effective_turn, previous_probs

    def predict_conversion(
        self,
        conversation_history: List[Dict[str, str]],
        conversation_id: str,
        is_incremental_prediction: bool = False
    ) -> Dict[str, Any]:
        """Predict conversion probability for a conversation."""

        normalized_history = conversation_history 

        effective_turn, previous_probs = self._get_effective_turn_for_prediction(
            normalized_history,
            conversation_id,
            is_incremental_prediction
        )
        
        logger.info(f"Predicting for conversation_id '{conversation_id}' at effective_turn: {effective_turn} (0-indexed).")

        full_text = " ".join([msg['message'] for msg in normalized_history])
        if not full_text.strip(): 
            logger.warning(f"Empty conversation for ID '{conversation_id}'. Using zero embedding.")
            embedding = np.zeros(self.expected_embedding_dim, dtype=np.float32)
        else:
            embedding = self.embedding_provider.get_embedding(full_text, effective_turn)

        metrics = self.embedding_provider.analyze_metrics(normalized_history, effective_turn)
        
        if 'outcome' not in metrics: 
            logger.error("'outcome' metric missing from provider. Defaulting to 0.5.")
            metrics['outcome'] = 0.5

        state_obj = ConversationState( 
            conversation_history=normalized_history,
            embedding=embedding,
            conversation_metrics=metrics, 
            turn_number=effective_turn, 
            conversion_probabilities=previous_probs 
        )

        observation = state_obj.state_vector
        
        expected_shape = self.model.observation_space.shape
        if observation.shape[0] != expected_shape[0]:
            logger.error(
                f"Observation shape mismatch for PPO model! Expected ({expected_shape[0]},), got ({observation.shape[0]},). "
                f"Effective_turn: {effective_turn}, Embedding shape: {embedding.shape}, "
                f"Metrics keys: {list(metrics.keys())}, Num prev_probs: {len(previous_probs)}"
            )
            raise ValueError("Observation shape mismatch. Cannot proceed with PPO model prediction.")

        action_raw, _ = self.model.predict(observation.astype(np.float32), deterministic=True)
        probability = float(np.clip(action_raw[0], 0.0, 1.0))

        updated_probs_for_state = previous_probs + [probability]
        self.conversation_states[conversation_id] = {
            'probabilities': updated_probs_for_state[-10:], 
            'turn_number': effective_turn + 1 
        }
        logger.debug(f"Updated state for conv_id '{conversation_id}': next turn {effective_turn + 1}, prev_probs count {len(updated_probs_for_state[-10:])}")


        return {
            'probability': probability,
            'turn': effective_turn, 
            'metrics': metrics, 
            'status': self._get_status(probability),
            'suggested_action': self._get_suggested_action(probability, metrics)
        }

    def generate_response_and_predict(
        self,
        conversation_history: List[Dict[str, str]], 
        user_input: str,
        conversation_id: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate sales response and then predict conversion probability."""
        
        response_text = self.embedding_provider.generate_response(
            history=conversation_history, 
            user_input=user_input,
            system_prompt=system_prompt
        )

        updated_conversation_history = conversation_history + [
            {'speaker': 'customer', 'message': user_input},
            {'speaker': 'sales_rep', 'message': response_text}
        ]
        
        prediction_result = self.predict_conversion(
            updated_conversation_history,
            conversation_id,
            is_incremental_prediction=False 
        )

        return {
            'response': response_text,
            'prediction': prediction_result
        }

    def _get_status(self, probability: float) -> str:
        if probability >= 0.7: return "🟢 High"
        if probability >= 0.5: return "🟡 Medium"
        if probability >= 0.3: return "🟠 Low"
        return "🔴 Very Low"

    def _get_suggested_action(self, probability: float, metrics: Dict[str, float]) -> str:
        cust_eng = metrics.get('customer_engagement', 0.5)
        sales_eff = metrics.get('sales_effectiveness', 0.5)

        if probability >= 0.7: 
            return "Focus on closing: Propose next steps, clarify final questions, or initiate purchase process."
        if probability >= 0.5:
            if sales_eff < 0.6:
                return "Address concerns and build value: Refine sales approach, highlight benefits."
            return "Build value: Reinforce benefits, handle objections, guide towards commitment."
        if probability >= 0.3:
            if cust_eng < 0.6:
                return "Re-engage customer: Ask open-ended questions, understand disengagement."
            return "Discover needs: Focus on deeper understanding of customer pain points."
        
        if cust_eng < 0.4 and sales_eff < 0.4:
            return "Re-qualify lead: Assess fit, identify misalignment, or consider disengaging."
        return "Identify barriers: Explore fundamental objections or lack of fit."