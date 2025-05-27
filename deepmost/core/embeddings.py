"""Embedding providers for different backends"""

import numpy as np
import torch
import logging
from typing import List, Dict, Optional, Protocol, Tuple, Any
from transformers import AutoTokenizer, AutoModel
import re
import json
import os
import random

logger = logging.getLogger(__name__)

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers"""

    def get_embedding(self, text: str, turn_number: int) -> np.ndarray:
        """Get embedding for text"""
        ...

    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, Any]:
        """Analyze conversation metrics"""
        ...

    def generate_response(
        self,
        history: List[Dict[str, str]],
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response (optional method)"""
        return "Thank you for your message. Could you tell me more?"


class OpenSourceEmbeddings:
    """Open-source embedding provider using HuggingFace models and LLM for comprehensive metrics analysis."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        expected_dim: int,
        llm_model: Optional[str] = None
    ):
        self.device = device
        self.expected_dim = expected_dim
        self.MAX_TURNS_REFERENCE = 1000

        logger.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.native_dim = self.model.config.hidden_size
        logger.info(f"Embedding model loaded. Native dim: {self.native_dim}, Expected dim: {self.expected_dim}")

        self.llm = None
        if llm_model:
            logger.info(f"Attempting to load GGUF LLM: {llm_model}")
            try:
                from llama_cpp import Llama
                
                llama_params = {
                    "n_gpu_layers": -1 if device.type == 'cuda' else 0,
                    "n_ctx": 8192,
                    "verbose": False 
                }

                if "/" in llm_model and not llm_model.lower().endswith(".gguf"):
                    repo_id = llm_model
                    logger.info(f"LLM '{repo_id}' is a HuggingFace repo. Using Llama.from_pretrained.")
                    
                    gguf_filename_pattern = "*Q4_K_M.gguf"
                    
                    try:
                        self.llm = Llama.from_pretrained(
                            repo_id=repo_id,
                            filename=gguf_filename_pattern, 
                            local_dir_use_symlinks=False, 
                            **llama_params
                        )
                        logger.info(f"LLM loaded successfully from HuggingFace repo '{repo_id}'.")

                    except Exception as e_from_pretrained:
                        logger.warning(f"Llama.from_pretrained failed for '{repo_id}' (pattern: '{gguf_filename_pattern}'): {e_from_pretrained}. "
                                       f"Ensure repo has a matching GGUF or try filename=None.")
                        raise 

                elif llm_model.lower().endswith(".gguf"): 
                    if not os.path.exists(llm_model):
                        logger.error(f"Local GGUF file not found: {llm_model}")
                        raise FileNotFoundError(f"Local GGUF file not found: {llm_model}")
                    logger.info(f"Loading LLM from local GGUF path: {llm_model}")
                    self.llm = Llama(model_path=llm_model, **llama_params)
                    logger.info(f"LLM loaded successfully from local path: {llm_model}")
                else:
                    logger.warning(f"LLM path '{llm_model}' not recognized as HF repo ID or local .gguf file. LLM not loaded.")
            
            except ImportError:
                logger.warning("llama-cpp-python is not installed. LLM features will be unavailable.")
            except FileNotFoundError as e: 
                logger.error(e)
                self.llm = None
            except Exception as e:
                logger.warning(f"Failed to load GGUF LLM '{llm_model}': {e}. LLM features may be unavailable.")
                self.llm = None
        
        if not self.llm:
            logger.info(
                "No LLM loaded. Metric analysis will use intelligent fallbacks. "
                "LLM-derived comprehensive metrics are highly recommended for best accuracy."
            )

    def get_embedding(self, text: str, turn_number: int) -> np.ndarray:
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors='pt', max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            embedding_native = normalized.cpu().numpy()[0]

        if embedding_native.shape[0] == self.expected_dim:
            embedding = embedding_native
        elif embedding_native.shape[0] > self.expected_dim:
            embedding = embedding_native[:self.expected_dim]
        else: 
            embedding = np.zeros(self.expected_dim, dtype=np.float32)
            embedding[:embedding_native.shape[0]] = embedding_native

        progress = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        scaled_embedding = embedding * (0.6 + 0.4 * progress)
        return scaled_embedding.astype(np.float32)

    def _get_comprehensive_metrics_from_llm(self, history: List[Dict[str, str]], turn_number: int) -> Tuple[Dict, bool]:
        """Get all sophisticated metrics from LLM via comprehensive JSON analysis."""
        llm_successfully_used = False
        if not self.llm:
            return self._get_fallback_metrics(history, turn_number), llm_successfully_used
        
        conversation_text = "\n".join([f"{msg['speaker'].capitalize()}: {msg['message']}" for msg in history])
        
        if not conversation_text.strip():
            logger.warning("Conversation history is empty for LLM comprehensive analysis. Using fallback.")
            return self._get_fallback_metrics(history, turn_number), llm_successfully_used

        prompt = f"""Analyze the following sales conversation and provide a comprehensive analysis in JSON format.

CONVERSATION:
---
{conversation_text}
---

Provide a detailed analysis covering ALL aspects below. Respond ONLY with valid JSON containing these exact keys:

{{
  "customer_engagement": 0.0-1.0,
  "sales_effectiveness": 0.0-1.0,
  "conversation_style": "string",
  "conversation_flow": "string", 
  "communication_channel": "string",
  "primary_customer_needs": ["list", "of", "needs"],
  "engagement_trend": 0.0-1.0,
  "objection_count": 0.0-1.0,
  "value_proposition_mentions": 0.0-1.0,
  "technical_depth": 0.0-1.0,
  "urgency_level": 0.0-1.0,
  "competitive_context": 0.0-1.0,
  "pricing_sensitivity": 0.0-1.0,
  "decision_authority_signals": 0.0-1.0
}}

CRITICAL: Respond with ONLY the JSON object. No explanations or additional text."""

        try:
            llm_response = self.llm(
                prompt,
                max_tokens=450,
                temperature=0.1,
                stop=["\n\n", "```"],
            )
            raw_llm_output = llm_response['choices'][0]['text'].strip()

            json_match = re.search(r"\{.*\}", raw_llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    validated_metrics = self._validate_and_normalize_metrics(parsed_json)
                    logger.info(f"Successfully parsed comprehensive LLM metrics with {len(validated_metrics)} fields")
                    llm_successfully_used = True
                    return validated_metrics, llm_successfully_used
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON from LLM output: '{json_str}'. Error: {e}. Using fallback.")
            else:
                logger.warning(f"No JSON object found in LLM output: '{raw_llm_output}'. Using fallback.")
        
        except Exception as e:
            logger.error(f"LLM comprehensive metrics analysis failed: {e}. Using fallback.", exc_info=True)
        
        return self._get_fallback_metrics(history, turn_number), llm_successfully_used

    def _validate_and_normalize_metrics(self, parsed_json: Dict) -> Dict:
        """Validate and normalize the LLM-provided metrics."""
        validated = {}
        
        numeric_fields = [
            'customer_engagement', 'sales_effectiveness', 'engagement_trend',
            'objection_count', 'value_proposition_mentions', 'technical_depth',
            'urgency_level', 'competitive_context', 'pricing_sensitivity',
            'decision_authority_signals'
        ]
        
        for field in numeric_fields:
            value = parsed_json.get(field)
            if isinstance(value, (int, float)):
                validated[field] = float(np.clip(value, 0.0, 1.0))
            else:
                validated[field] = 0.5 
        
        validated['conversation_style'] = parsed_json.get('conversation_style', 'direct_professional')
        validated['conversation_flow'] = parsed_json.get('conversation_flow', 'standard_linear')
        validated['communication_channel'] = parsed_json.get('communication_channel', 'email')
        
        needs = parsed_json.get('primary_customer_needs', ['efficiency', 'cost_reduction'])
        if isinstance(needs, list):
            validated['primary_customer_needs'] = needs[:3]
        else:
            validated['primary_customer_needs'] = ['efficiency', 'cost_reduction']
        
        return validated

    def _get_fallback_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict:
        """Generate intelligent fallback metrics when LLM is not available or fails."""
        customer_text = " ".join([msg['message'].lower() for msg in history if msg['speaker'] == 'customer'])

        return {
            'customer_engagement': 0.5,
            'sales_effectiveness': 0.5,
            'conversation_style': 'direct_professional',
            'conversation_flow': 'standard_linear',
            'communication_channel': 'email',
            'primary_customer_needs': ['efficiency', 'cost_reduction'],
            'engagement_trend': 0.5,
            'objection_count': 0.4 if any(obj in customer_text for obj in ['expensive', 'costly', 'concern', 'budget', 'not interested', 'problem', 'issue']) else 0.1,
            'value_proposition_mentions': 0.4,
            'technical_depth': 0.3,
            'urgency_level': 0.2,
            'competitive_context': 0.1,
            'pricing_sensitivity': 0.5 if any(kw in customer_text for kw in ['price', 'cost', 'budget', 'expensive']) else 0.2,
            'decision_authority_signals': 0.5
        }

    def _generate_probability_trajectory(self, history: List[Dict[str, str]], base_metrics: Dict) -> Dict[int, float]:
        """Generate realistic probability trajectory using comprehensive metrics."""
        trajectory = {}
        num_turns = len(history)
        
        if num_turns == 0:
            return {0: 0.5}
        
        engagement = base_metrics.get('customer_engagement', 0.5)
        effectiveness = base_metrics.get('sales_effectiveness', 0.5)
        engagement_trend = base_metrics.get('engagement_trend', 0.5)
        objection_level = base_metrics.get('objection_count', 0.3)
        
        current_prob = 0.15
        
        for i in range(num_turns):
            turn_factor = 0.0
            turn_factor += (engagement - 0.5) * 0.2
            turn_factor += (effectiveness - 0.5) * 0.15
            turn_factor -= objection_level * 0.25
            
            if engagement_trend > 0.7:
                turn_factor += 0.05 * (i / max(1, num_turns -1))
            elif engagement_trend < 0.3:
                turn_factor -= 0.05 * (i / max(1, num_turns -1))

            max_delta = 0.15 
            delta = np.clip(turn_factor, -max_delta, max_delta)
            
            current_prob += delta
            current_prob += random.uniform(-0.03, 0.03)
            current_prob = np.clip(current_prob, 0.05, 0.95)
            trajectory[i] = round(current_prob, 4)
            
            engagement = np.clip(engagement + (engagement_trend - 0.5) * 0.05, 0, 1)

        return trajectory

    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, Any]:
        conversation_length = float(len(history))
        progress_metric = min(1.0, turn_number / self.MAX_TURNS_REFERENCE) if self.MAX_TURNS_REFERENCE > 0 else 0.0
        
        base_metrics, llm_data_was_successfully_used = self._get_comprehensive_metrics_from_llm(history, turn_number)
        probability_trajectory = self._generate_probability_trajectory(history, base_metrics)
        
        final_metrics = {
            'customer_engagement': base_metrics['customer_engagement'],
            'sales_effectiveness': base_metrics['sales_effectiveness'],
            'conversation_length': conversation_length,
            'outcome': 0.5,
            'progress': progress_metric,
            'conversation_style': base_metrics['conversation_style'],
            'conversation_flow': base_metrics['conversation_flow'],
            'communication_channel': base_metrics['communication_channel'],
            'primary_customer_needs': base_metrics['primary_customer_needs'],
            'probability_trajectory': probability_trajectory,
            'engagement_trend': base_metrics['engagement_trend'],
            'objection_count': base_metrics['objection_count'],
            'value_proposition_mentions': base_metrics['value_proposition_mentions'],
            'technical_depth': base_metrics['technical_depth'],
            'urgency_level': base_metrics['urgency_level'],
            'competitive_context': base_metrics['competitive_context'],
            'pricing_sensitivity': base_metrics['pricing_sensitivity'],
            'decision_authority_signals': base_metrics['decision_authority_signals']
        }
        
        logger.info(f"Comprehensive Metrics Analysis (LLM data successfully used: {llm_data_was_successfully_used}) - "
                   f"Engagement: {final_metrics['customer_engagement']:.2f}, "
                   f"Effectiveness: {final_metrics['sales_effectiveness']:.2f}")
        
        return final_metrics

    def generate_response(
        self,
        history: List[Dict[str, str]],
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> str:
        if not self.llm:
            logger.warning("LLM not available for response generation. Returning canned response.")
            return "Thank you for your message. Could you provide more details?"

        messages_for_llm = []
        if system_prompt:
            messages_for_llm.append({"role": "system", "content": system_prompt})

        for msg in history:
            role = "user" if msg['speaker'] == 'customer' else "assistant"
            messages_for_llm.append({"role": role, "content": msg['message']})
        
        messages_for_llm.append({"role": "user", "content": user_input})
        
        try:
            chat_completion = self.llm.create_chat_completion(
                messages=messages_for_llm,
                max_tokens=150,
                temperature=0.7,
                stop=["\nUser:", "\nCustomer:", "\n<|user|>", "\n<|end|>"] 
            )
            generated_text = chat_completion['choices'][0]['message']['content'].strip()
            logger.info(f"LLM generated response: {generated_text}")
            return generated_text
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}", exc_info=True)
            return "I understand. Could you please provide more details about what you're looking for?"


class AzureEmbeddings:
    """Azure OpenAI embedding provider with full chat completion support."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        embedding_deployment: str,
        chat_deployment: Optional[str] = None,
        api_version: str = "2024-10-21",
        expected_dim: int = 1536
    ):
        from openai import AzureOpenAI

        self.api_key = api_key
        self.endpoint = endpoint
        self.embedding_deployment = embedding_deployment
        self.chat_deployment = chat_deployment
        self.api_version = api_version
        self.expected_dim = expected_dim
        self.native_dim = 0
        self.MAX_TURNS_REFERENCE = 1000

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )

        # Test embedding connection
        try:
            logger.info(f"Testing Azure OpenAI embedding connection with deployment: {self.embedding_deployment}")
            test_response = self.client.embeddings.create(
                input="test",
                model=self.embedding_deployment
            )
            self.native_dim = len(test_response.data[0].embedding)
            logger.info(f"Azure embeddings initialized successfully. Native dim: {self.native_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure embeddings: {e}")
            raise

        # Test chat connection if deployment provided
        self.chat_available = False
        if self.chat_deployment:
            try:
                test_chat_response = self.client.chat.completions.create(
                    model=self.chat_deployment,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                self.chat_available = True
                logger.info(f"Azure chat completions initialized successfully with deployment: {self.chat_deployment}")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure chat completions: {e}")
                self.chat_available = False
        else:
            logger.info("No chat deployment provided. LLM-powered metrics will be unavailable.")

    def get_embedding(self, text: str, turn_number: int) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            embedding_native = np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Azure embedding API call failed: {e}")
            embedding_native = np.zeros(self.expected_dim, dtype=np.float32)

        if embedding_native.shape[0] == self.expected_dim:
            embedding = embedding_native
        elif embedding_native.shape[0] > self.expected_dim:
            embedding = embedding_native[:self.expected_dim] 
        else: 
            embedding = np.zeros(self.expected_dim, dtype=np.float32)
            embedding[:embedding_native.shape[0]] = embedding_native 

        progress = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        scaled_embedding = embedding * (0.6 + 0.4 * progress)
        return scaled_embedding.astype(np.float32)

    def _get_comprehensive_metrics_from_azure_llm(self, history: List[Dict[str, str]], turn_number: int) -> Tuple[Dict, bool]:
        """Get comprehensive metrics from Azure OpenAI chat completions."""
        azure_llm_successfully_used = False
        
        if not self.chat_available:
            return self._get_fallback_metrics(history, turn_number), azure_llm_successfully_used
        
        conversation_text = "\n".join([f"{msg['speaker'].capitalize()}: {msg['message']}" for msg in history])
        
        if not conversation_text.strip():
            return self._get_fallback_metrics(history, turn_number), azure_llm_successfully_used

        system_prompt = """You are an expert sales conversation analyst. Analyze conversations and provide detailed metrics in JSON format. Always respond with ONLY valid JSON containing the exact keys requested."""
        
        user_prompt = f"""Analyze the following sales conversation and provide JSON with metrics for customer_engagement, sales_effectiveness, conversation_style, conversation_flow, communication_channel, primary_customer_needs, engagement_trend, objection_count, value_proposition_mentions, technical_depth, urgency_level, competitive_context, pricing_sensitivity, and decision_authority_signals.

CONVERSATION:
{conversation_text}

Respond with ONLY the JSON object."""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            raw_azure_output = response.choices[0].message.content.strip()

            json_match = re.search(r"\{.*\}", raw_azure_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    validated_metrics = self._validate_and_normalize_metrics(parsed_json)
                    logger.info(f"Successfully parsed comprehensive Azure LLM metrics")
                    azure_llm_successfully_used = True
                    return validated_metrics, azure_llm_successfully_used
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON from Azure LLM output. Using fallback.")
            else:
                logger.warning(f"No JSON object found in Azure LLM output. Using fallback.")
        
        except Exception as e:
            logger.error(f"Azure LLM comprehensive metrics analysis failed: {e}. Using fallback.")
        
        return self._get_fallback_metrics(history, turn_number), azure_llm_successfully_used

    def _validate_and_normalize_metrics(self, parsed_json: Dict) -> Dict:
        """Validate and normalize the Azure LLM-provided metrics."""
        validated = {}
        
        numeric_fields = [
            'customer_engagement', 'sales_effectiveness', 'engagement_trend',
            'objection_count', 'value_proposition_mentions', 'technical_depth',
            'urgency_level', 'competitive_context', 'pricing_sensitivity',
            'decision_authority_signals'
        ]
        
        for field in numeric_fields:
            value = parsed_json.get(field)
            if isinstance(value, (int, float)):
                validated[field] = float(np.clip(value, 0.0, 1.0))
            else:
                validated[field] = 0.5
        
        validated['conversation_style'] = parsed_json.get('conversation_style', 'direct_professional')
        validated['conversation_flow'] = parsed_json.get('conversation_flow', 'standard_linear')
        validated['communication_channel'] = parsed_json.get('communication_channel', 'email')
        
        needs = parsed_json.get('primary_customer_needs', ['efficiency', 'cost_reduction'])
        if isinstance(needs, list):
            validated['primary_customer_needs'] = needs[:3]
        else:
            validated['primary_customer_needs'] = ['efficiency', 'cost_reduction']
        
        return validated

    def _get_fallback_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict:
        """Generate intelligent fallback metrics when Azure LLM is not available."""        
        customer_text = " ".join([msg['message'].lower() for msg in history if msg['speaker'] == 'customer'])
        
        engagement = 0.5
        effectiveness = 0.5
        
        if any(signal in customer_text for signal in ['buy', 'purchase', 'interested', 'yes', 'great', 'sounds good']):
            engagement = 0.7
        if any(obj in customer_text for obj in ['expensive', 'costly', 'not interested', 'no', 'concern', 'problem']):
            engagement = 0.3

        return {
            'customer_engagement': engagement,
            'sales_effectiveness': effectiveness,
            'conversation_style': 'direct_professional',
            'conversation_flow': 'standard_linear',
            'communication_channel': 'email',
            'primary_customer_needs': ['efficiency', 'cost_reduction'],
            'engagement_trend': 0.5,
            'objection_count': 0.3 if any(obj in customer_text for obj in ['expensive', 'costly', 'concern', 'not interested']) else 0.1,
            'value_proposition_mentions': 0.3,
            'technical_depth': 0.4,
            'urgency_level': 0.2,
            'competitive_context': 0.1,
            'pricing_sensitivity': 0.4 if any(kw in customer_text for kw in ['price', 'cost', 'budget']) else 0.2,
            'decision_authority_signals': 0.5
        }

    def _generate_probability_trajectory(self, history: List[Dict[str, str]], base_metrics: Dict) -> Dict[int, float]:
        """Generate realistic probability trajectory using Azure comprehensive metrics."""
        trajectory = {}
        num_turns = len(history)
        
        if num_turns == 0:
            return {0: 0.5}
        
        engagement = base_metrics.get('customer_engagement', 0.5)
        effectiveness = base_metrics.get('sales_effectiveness', 0.5)
        engagement_trend = base_metrics.get('engagement_trend', 0.5)
        objection_level = base_metrics.get('objection_count', 0.3)
        
        current_prob = 0.2
        
        for i in range(num_turns):
            turn_factor = 0.0
            turn_factor += (engagement - 0.5) * 0.2
            turn_factor += (effectiveness - 0.5) * 0.15
            turn_factor -= objection_level * 0.25
            
            if engagement_trend > 0.7:
                turn_factor += 0.05 * (i / max(1, num_turns - 1))
            elif engagement_trend < 0.3:
                turn_factor -= 0.05 * (i / max(1, num_turns - 1))

            max_delta = 0.15
            delta = np.clip(turn_factor, -max_delta, max_delta)
            
            current_prob += delta
            current_prob += random.uniform(-0.02, 0.02)
            current_prob = np.clip(current_prob, 0.05, 0.95)
            trajectory[i] = round(current_prob, 4)

        return trajectory

    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, Any]:
        conversation_length = float(len(history))
        progress_metric = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        
        base_metrics, azure_llm_data_was_successfully_used = self._get_comprehensive_metrics_from_azure_llm(history, turn_number)
        probability_trajectory = self._generate_probability_trajectory(history, base_metrics)
        
        final_metrics = {
            'customer_engagement': base_metrics['customer_engagement'],
            'sales_effectiveness': base_metrics['sales_effectiveness'],
            'conversation_length': conversation_length,
            'outcome': 0.5,
            'progress': progress_metric,
            'conversation_style': base_metrics['conversation_style'],
            'conversation_flow': base_metrics['conversation_flow'],
            'communication_channel': base_metrics['communication_channel'],
            'primary_customer_needs': base_metrics['primary_customer_needs'],
            'probability_trajectory': probability_trajectory,
            'engagement_trend': base_metrics['engagement_trend'],
            'objection_count': base_metrics['objection_count'],
            'value_proposition_mentions': base_metrics['value_proposition_mentions'],
            'technical_depth': base_metrics['technical_depth'],
            'urgency_level': base_metrics['urgency_level'],
            'competitive_context': base_metrics['competitive_context'],
            'pricing_sensitivity': base_metrics['pricing_sensitivity'],
            'decision_authority_signals': base_metrics['decision_authority_signals']
        }
        
        backend_type = "Azure LLM" if azure_llm_data_was_successfully_used else "Azure Fallback"
        logger.info(f"Comprehensive Metrics Analysis ({backend_type}) - "
                   f"Engagement: {final_metrics['customer_engagement']:.2f}")
        
        return final_metrics

    def generate_response(
        self,
        history: List[Dict[str, str]],
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> str:
        if not self.chat_available:
            logger.warning("Azure chat completions not available. Returning enhanced canned response.")
            return "Thank you for your inquiry. I understand your interest and would be happy to help. Could you provide more details about your specific needs?"

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            default_system_prompt = """You are a professional sales representative. Provide helpful, informative responses that build value and address customer needs."""
            messages.append({"role": "system", "content": default_system_prompt})
        
        for msg in history:
            role = "user" if msg['speaker'] == 'customer' else "assistant"
            messages.append({"role": role, "content": msg['message']})
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_response = response.choices[0].message.content.strip()
            logger.info(f"Azure OpenAI generated response: {generated_response}")
            return generated_response
            
        except Exception as e:
            logger.error(f"Azure chat completion failed: {e}")
            return "I appreciate your message. Let me help you find the right solution. Could you tell me more about what you're looking for?"


class OpenAIEmbeddings:
    """Standard OpenAI embedding provider with full chat completion support."""

    def __init__(
        self,
        api_key: str,
        embedding_model: str = "text-embedding-3-large",
        chat_model: Optional[str] = None,
        expected_dim: int = 3072
    ):
        from openai import OpenAI

        self.api_key = api_key
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.expected_dim = expected_dim
        self.native_dim = 0
        self.MAX_TURNS_REFERENCE = 1000

        self.client = OpenAI(api_key=api_key)

        # Test embedding connection
        try:
            logger.info(f"Testing OpenAI embedding connection with model: {self.embedding_model}")
            test_response = self.client.embeddings.create(
                input="test",
                model=self.embedding_model
            )
            self.native_dim = len(test_response.data[0].embedding)
            logger.info(f"OpenAI embeddings initialized successfully. Native dim: {self.native_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise

        # Test chat connection if model provided
        self.chat_available = False
        if self.chat_model:
            try:
                test_chat_response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                self.chat_available = True
                logger.info(f"OpenAI chat completions initialized successfully with model: {self.chat_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI chat completions: {e}")
                self.chat_available = False
        else:
            logger.info("No chat model provided. LLM-powered metrics will be unavailable.")

    def get_embedding(self, text: str, turn_number: int) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding_native = np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"OpenAI embedding API call failed: {e}")
            embedding_native = np.zeros(self.expected_dim, dtype=np.float32)

        if embedding_native.shape[0] == self.expected_dim:
            embedding = embedding_native
        elif embedding_native.shape[0] > self.expected_dim:
            embedding = embedding_native[:self.expected_dim] 
        else: 
            embedding = np.zeros(self.expected_dim, dtype=np.float32)
            embedding[:embedding_native.shape[0]] = embedding_native 

        progress = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        scaled_embedding = embedding * (0.6 + 0.4 * progress)
        return scaled_embedding.astype(np.float32)

    def _get_comprehensive_metrics_from_openai_llm(self, history: List[Dict[str, str]], turn_number: int) -> Tuple[Dict, bool]:
        """Get comprehensive metrics from OpenAI chat completions."""
        openai_llm_successfully_used = False
        
        if not self.chat_available:
            return self._get_fallback_metrics(history, turn_number), openai_llm_successfully_used
        
        conversation_text = "\n".join([f"{msg['speaker'].capitalize()}: {msg['message']}" for msg in history])
        
        if not conversation_text.strip():
            return self._get_fallback_metrics(history, turn_number), openai_llm_successfully_used

        system_prompt = """You are an expert sales conversation analyst. Analyze conversations and provide detailed metrics in JSON format. Always respond with ONLY valid JSON containing the exact keys requested."""
        
        user_prompt = f"""Analyze the following sales conversation and provide JSON with metrics for customer_engagement, sales_effectiveness, conversation_style, conversation_flow, communication_channel, primary_customer_needs, engagement_trend, objection_count, value_proposition_mentions, technical_depth, urgency_level, competitive_context, pricing_sensitivity, and decision_authority_signals.

CONVERSATION:
{conversation_text}

Respond with ONLY the JSON object."""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            raw_openai_output = response.choices[0].message.content.strip()

            json_match = re.search(r"\{.*\}", raw_openai_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    validated_metrics = self._validate_and_normalize_metrics(parsed_json)
                    logger.info(f"Successfully parsed comprehensive OpenAI LLM metrics")
                    openai_llm_successfully_used = True
                    return validated_metrics, openai_llm_successfully_used
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON from OpenAI LLM output. Using fallback.")
            else:
                logger.warning(f"No JSON object found in OpenAI LLM output. Using fallback.")
        
        except Exception as e:
            logger.error(f"OpenAI LLM comprehensive metrics analysis failed: {e}. Using fallback.")
        
        return self._get_fallback_metrics(history, turn_number), openai_llm_successfully_used

    def _validate_and_normalize_metrics(self, parsed_json: Dict) -> Dict:
        """Validate and normalize the OpenAI LLM-provided metrics."""
        validated = {}
        
        numeric_fields = [
            'customer_engagement', 'sales_effectiveness', 'engagement_trend',
            'objection_count', 'value_proposition_mentions', 'technical_depth',
            'urgency_level', 'competitive_context', 'pricing_sensitivity',
            'decision_authority_signals'
        ]
        
        for field in numeric_fields:
            value = parsed_json.get(field)
            if isinstance(value, (int, float)):
                validated[field] = float(np.clip(value, 0.0, 1.0))
            else:
                validated[field] = 0.5
        
        validated['conversation_style'] = parsed_json.get('conversation_style', 'direct_professional')
        validated['conversation_flow'] = parsed_json.get('conversation_flow', 'standard_linear')
        validated['communication_channel'] = parsed_json.get('communication_channel', 'email')
        
        needs = parsed_json.get('primary_customer_needs', ['efficiency', 'cost_reduction'])
        if isinstance(needs, list):
            validated['primary_customer_needs'] = needs[:3]
        else:
            validated['primary_customer_needs'] = ['efficiency', 'cost_reduction']
        
        return validated

    def _get_fallback_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict:
        """Generate intelligent fallback metrics when OpenAI LLM is not available."""        
        customer_text = " ".join([msg['message'].lower() for msg in history if msg['speaker'] == 'customer'])
        
        engagement = 0.5
        effectiveness = 0.5
        
        if any(signal in customer_text for signal in ['buy', 'purchase', 'interested', 'yes', 'great', 'sounds good']):
            engagement = 0.7
        if any(obj in customer_text for obj in ['expensive', 'costly', 'not interested', 'no', 'concern', 'problem']):
            engagement = 0.3

        return {
            'customer_engagement': engagement,
            'sales_effectiveness': effectiveness,
            'conversation_style': 'direct_professional',
            'conversation_flow': 'standard_linear',
            'communication_channel': 'email',
            'primary_customer_needs': ['efficiency', 'cost_reduction'],
            'engagement_trend': 0.5,
            'objection_count': 0.3 if any(obj in customer_text for obj in ['expensive', 'costly', 'concern', 'not interested']) else 0.1,
            'value_proposition_mentions': 0.3,
            'technical_depth': 0.4,
            'urgency_level': 0.2,
            'competitive_context': 0.1,
            'pricing_sensitivity': 0.4 if any(kw in customer_text for kw in ['price', 'cost', 'budget']) else 0.2,
            'decision_authority_signals': 0.5
        }

    def _generate_probability_trajectory(self, history: List[Dict[str, str]], base_metrics: Dict) -> Dict[int, float]:
        """Generate realistic probability trajectory using OpenAI comprehensive metrics."""
        trajectory = {}
        num_turns = len(history)
        
        if num_turns == 0:
            return {0: 0.5}
        
        engagement = base_metrics.get('customer_engagement', 0.5)
        effectiveness = base_metrics.get('sales_effectiveness', 0.5)
        engagement_trend = base_metrics.get('engagement_trend', 0.5)
        objection_level = base_metrics.get('objection_count', 0.3)
        
        current_prob = 0.25  # Slightly higher starting point for OpenAI
        
        for i in range(num_turns):
            turn_factor = 0.0
            turn_factor += (engagement - 0.5) * 0.2
            turn_factor += (effectiveness - 0.5) * 0.15
            turn_factor -= objection_level * 0.25
            
            if engagement_trend > 0.7:
                turn_factor += 0.05 * (i / max(1, num_turns - 1))
            elif engagement_trend < 0.3:
                turn_factor -= 0.05 * (i / max(1, num_turns - 1))

            max_delta = 0.15
            delta = np.clip(turn_factor, -max_delta, max_delta)
            
            current_prob += delta
            current_prob += random.uniform(-0.02, 0.02)
            current_prob = np.clip(current_prob, 0.05, 0.95)
            trajectory[i] = round(current_prob, 4)

        return trajectory

    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, Any]:
        conversation_length = float(len(history))
        progress_metric = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        
        base_metrics, openai_llm_data_was_successfully_used = self._get_comprehensive_metrics_from_openai_llm(history, turn_number)
        probability_trajectory = self._generate_probability_trajectory(history, base_metrics)
        
        final_metrics = {
            'customer_engagement': base_metrics['customer_engagement'],
            'sales_effectiveness': base_metrics['sales_effectiveness'],
            'conversation_length': conversation_length,
            'outcome': 0.5,
            'progress': progress_metric,
            'conversation_style': base_metrics['conversation_style'],
            'conversation_flow': base_metrics['conversation_flow'],
            'communication_channel': base_metrics['communication_channel'],
            'primary_customer_needs': base_metrics['primary_customer_needs'],
            'probability_trajectory': probability_trajectory,
            'engagement_trend': base_metrics['engagement_trend'],
            'objection_count': base_metrics['objection_count'],
            'value_proposition_mentions': base_metrics['value_proposition_mentions'],
            'technical_depth': base_metrics['technical_depth'],
            'urgency_level': base_metrics['urgency_level'],
            'competitive_context': base_metrics['competitive_context'],
            'pricing_sensitivity': base_metrics['pricing_sensitivity'],
            'decision_authority_signals': base_metrics['decision_authority_signals']
        }
        
        backend_type = "OpenAI LLM" if openai_llm_data_was_successfully_used else "OpenAI Fallback"
        logger.info(f"Comprehensive Metrics Analysis ({backend_type}) - "
                   f"Engagement: {final_metrics['customer_engagement']:.2f}")
        
        return final_metrics

    def generate_response(
        self,
        history: List[Dict[str, str]],
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> str:
        if not self.chat_available:
            logger.warning("OpenAI chat completions not available. Returning enhanced canned response.")
            return "Thank you for your inquiry. I understand your interest and would be happy to help. Could you provide more details about your specific needs?"

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            default_system_prompt = """You are a professional sales representative. Provide helpful, informative responses that build value and address customer needs."""
            messages.append({"role": "system", "content": default_system_prompt})
        
        for msg in history:
            role = "user" if msg['speaker'] == 'customer' else "assistant"
            messages.append({"role": role, "content": msg['message']})
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_response = response.choices[0].message.content.strip()
            logger.info(f"OpenAI generated response: {generated_response}")
            return generated_response
            
        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {e}")
            return "I appreciate your message. Let me help you find the right solution. Could you tell me more about what you're looking for?"