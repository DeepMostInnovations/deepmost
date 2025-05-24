"""Embedding providers for different backends"""

import numpy as np
import torch
import logging
from typing import List, Dict, Optional, Protocol
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers"""
    
    def get_embedding(self, text: str, turn_number: int) -> np.ndarray:
        """Get embedding for text"""
        ...
    
    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, float]:
        """Analyze conversation metrics"""
        ...


class OpenSourceEmbeddings:
    """Open-source embedding provider using HuggingFace models"""
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        expected_dim: int,
        llm_model: Optional[str] = None
    ):
        self.device = device
        self.expected_dim = expected_dim
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.native_dim = self.model.config.hidden_size
        
        # Load LLM if provided
        self.llm = None
        if llm_model:
            try:
                from llama_cpp import Llama
                if "/" in llm_model:  # HF repo
                    self.llm = Llama.from_pretrained(
                        repo_id=llm_model,
                        filename="*Q4_K_M.gguf",
                        n_gpu_layers=-1 if device.type == 'cuda' else 0,
                        n_ctx=2048,
                        verbose=False
                    )
                else:  # Local file
                    self.llm = Llama(
                        model_path=llm_model,
                        n_gpu_layers=-1 if device.type == 'cuda' else 0,
                        n_ctx=2048,
                        verbose=False
                    )
            except Exception as e:
                logger.warning(f"Failed to load LLM: {e}")
    
    def get_embedding(self, text: str, turn_number: int) -> np.ndarray:
        """Get embedding with turn-based scaling"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            # Normalize
            normalized = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            embedding = normalized.cpu().numpy()[0]
        
        # Adjust dimension
        if embedding.shape[0] != self.expected_dim:
            adjusted = np.zeros(self.expected_dim, dtype=np.float32)
            copy_len = min(embedding.shape[0], self.expected_dim)
            adjusted[:copy_len] = embedding[:copy_len]
            embedding = adjusted
        
        # Apply turn-based scaling
        progress = min(1.0, turn_number / 20)
        scaled = embedding * (0.6 + 0.4 * progress)
        
        return scaled.astype(np.float32)
    
    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, float]:
        """Analyze metrics using LLM if available"""
        base_metrics = {
            'customer_engagement': 0.5,
            'sales_effectiveness': 0.5,
            'conversation_length': float(len(history)),
            'outcome': 0.5,
            'progress': min(1.0, turn_number / 20)
        }
        
        if self.llm:
            # Use LLM to analyze
            conversation_text = "\n".join([f"{msg['speaker']}: {msg['message']}" for msg in history])
            prompt = f"""Analyze this sales conversation and provide scores:
customer_engagement: (0.0-1.0)
sales_effectiveness: (0.0-1.0)

{conversation_text}

Scores:"""
            
            try:
                response = self.llm(prompt, max_tokens=50, temperature=0.1)
                text = response['choices'][0]['text']
                
                # Parse scores
                import re
                for line in text.split('\n'):
                    if 'customer_engagement:' in line:
                        match = re.search(r'(\d\.\d+)', line)
                        if match:
                            base_metrics['customer_engagement'] = float(match.group(1))
                    elif 'sales_effectiveness:' in line:
                        match = re.search(r'(\d\.\d+)', line)
                        if match:
                            base_metrics['sales_effectiveness'] = float(match.group(1))
            except Exception as e:
                logger.debug(f"LLM metrics analysis failed: {e}")
        
        return base_metrics
    
    def generate_response(
        self,
        history: List[Dict[str, str]],
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using LLM"""
        if not self.llm:
            return "Thank you for your message. Could you tell me more about your specific needs?"
        
        # Format prompt
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"{system_prompt}\n\n")
        
        for msg in history:
            role = "user" if msg['speaker'] == 'customer' else "model"
            prompt_parts.append(f"<start_of_turn>{role}\n{msg['message']}<end_of_turn>")
        
        prompt_parts.append(f"<start_of_turn>user\n{user_input}<end_of_turn>")
        prompt_parts.append("<start_of_turn>model")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = self.llm(prompt, max_tokens=200, temperature=0.7)
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I understand. Could you please provide more details about what you're looking for?"


class AzureEmbeddings:
    """Azure OpenAI embedding provider"""
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str,
        expected_dim: int
    ):
        from openai import AzureOpenAI
        
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.deployment = deployment
        self.expected_dim = expected_dim
        
        # Test connection
        try:
            test_response = self.client.embeddings.create(
                input="test",
                model=deployment
            )
            self.native_dim = len(test_response.data[0].embedding)
            logger.info(f"Azure embeddings initialized. Native dim: {self.native_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure embeddings: {e}")
            raise
    
    def get_embedding(self, text: str, turn_number: int) -> np.ndarray:
        """Get embedding from Azure"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Azure embedding failed: {e}")
            embedding = np.zeros(self.native_dim, dtype=np.float32)
        
        # Apply turn-based scaling
        progress = min(1.0, turn_number / 20)
        scaled = embedding * (0.6 + 0.4 * progress)
        
        # Adjust dimension if needed
        if self.native_dim > self.expected_dim:
            # Mini-embedding logic
            pool_factor = self.native_dim // self.expected_dim
            pooled = scaled[:self.expected_dim * pool_factor].reshape(-1, pool_factor).mean(axis=1)
            return pooled.astype(np.float32)
        elif self.native_dim < self.expected_dim:
            # Pad
            padded = np.zeros(self.expected_dim, dtype=np.float32)
            padded[:self.native_dim] = scaled
            return padded
        else:
            return scaled.astype(np.float32)
    
    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, float]:
        """Basic metrics analysis"""
        return {
            'customer_engagement': 0.5,
            'sales_effectiveness': 0.5,
            'conversation_length': float(len(history)),
            'outcome': 0.5,
            'progress': min(1.0, turn_number / 20)
        }