"""Utility functions and classes"""

import os
import torch
import torch.nn as nn
import numpy as np
import requests
from tqdm import tqdm
from typing import List, Dict, Any
from dataclasses import dataclass
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


@dataclass
class ConversationState:
    """State representation for a conversation"""
    conversation_history: List[Dict[str, str]]
    embedding: np.ndarray
    conversation_metrics: Dict[str, float]
    turn_number: int
    conversion_probabilities: List[float]

    @property
    def state_vector(self) -> np.ndarray:
        """Create state vector for model input"""
        metric_values = np.array([
            self.conversation_metrics.get('customer_engagement', 0.5),
            self.conversation_metrics.get('sales_effectiveness', 0.5),
            self.conversation_metrics.get('conversation_length', 0.0),
            self.conversation_metrics.get('outcome', 0.5),
            self.conversation_metrics.get('progress', 0.0)
        ], dtype=np.float32)
        
        turn_info = np.array([float(self.turn_number)], dtype=np.float32)
        
        padded_probs = np.zeros(10, dtype=np.float32)
        if self.conversion_probabilities:
            recent_probs = self.conversion_probabilities[-10:]
            padded_probs[:len(recent_probs)] = recent_probs
        
        return np.concatenate([
            self.embedding,
            metric_values,
            turn_info,
            padded_probs
        ]).astype(np.float32)


class CustomLN(BaseFeaturesExtractor):
    """Custom feature extractor matching training architecture"""
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.linear_network = nn.Sequential(
            nn.Linear(n_input_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear_network(observations)


def download_model(url: str, dest_path: str):
    """Download model file with progress bar"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading model") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def normalize_conversation(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Normalize conversation format"""
    normalized = []
    
    for msg in history:
        # Handle different key names
        speaker = msg.get('speaker', msg.get('role', '')).lower()
        message = msg.get('message', msg.get('content', ''))
        
        # Map to standard speakers
        if speaker in ['user', 'customer']:
            speaker = 'customer'
        elif speaker in ['assistant', 'sales_rep', 'agent', 'bot', 'model']:
            speaker = 'sales_rep'
        
        normalized.append({
            'speaker': speaker,
            'message': message
        })
    
    return normalized