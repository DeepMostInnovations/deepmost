"""Embedding providers for different backends"""

import numpy as np
import torch
import logging
from typing import List, Dict, Optional, Protocol
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

    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, float]:
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
        self.MAX_TURNS_REFERENCE = 20

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

    def _get_comprehensive_metrics_from_llm(self, history: List[Dict[str, str]], turn_number: int) -> Dict:
        """Get all sophisticated metrics from LLM via comprehensive JSON analysis."""
        if not self.llm:
            return self._get_fallback_metrics(history, turn_number)
        
        conversation_text = "\n".join([f"{msg['speaker'].capitalize()}: {msg['message']}" for msg in history])
        
        if not conversation_text.strip():
            logger.warning("Conversation history is empty for LLM comprehensive analysis.")
            return self._get_fallback_metrics(history, turn_number)

        # Comprehensive prompt for all metrics
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

DETAILED SCORING GUIDELINES:

**customer_engagement** (0.0-1.0): How engaged and interested the customer appears
- 0.0-0.2: Completely disengaged, very short responses, no interest
- 0.3-0.4: Low engagement, minimal responses, seems distracted
- 0.5-0.6: Moderate engagement, participating normally
- 0.7-0.8: High engagement, asking questions, showing interest
- 0.9-1.0: Extremely engaged, ready to buy, making decisions

**sales_effectiveness** (0.0-1.0): How effective the sales representative's approach is
- 0.0-0.2: Very poor approach, pushing away customer, unprofessional
- 0.3-0.4: Poor approach, not addressing needs, unclear responses
- 0.5-0.6: Adequate approach, some value shown, decent communication
- 0.7-0.8: Good approach, addressing needs well, building value
- 0.9-1.0: Excellent approach, perfectly addressing needs, very professional

**conversation_style** (string): Choose the most fitting style:
"casual_friendly", "direct_professional", "technical_detailed", "consultative_advisory", 
"empathetic_supportive", "skeptical_challenging", "urgent_time_pressed", "confused_overwhelmed", 
"knowledgeable_assertive", "storytelling_narrative"

**conversation_flow** (string): Choose the most fitting flow pattern:
"standard_linear", "multiple_objection_loops", "subject_switching", "interrupted_followup", 
"technical_deep_dive", "competitive_comparison", "gradual_discovery", "immediate_interest", 
"initial_rejection", "stakeholder_expansion", "pricing_negotiation", "implementation_concerns", 
"value_justification", "relationship_building", "multi_session", "demo_walkthrough"

**communication_channel** (string): Choose the most likely channel:
"email", "live_chat", "phone_call", "video_call", "in_person", "sms", "social_media"

**primary_customer_needs** (array): Select 2-3 most relevant needs:
"efficiency", "cost_reduction", "growth", "compliance", "integration", "usability", 
"reliability", "security", "support", "analytics"

**engagement_trend** (0.0-1.0): Is customer engagement increasing or decreasing?
- 0.0-0.3: Strongly declining engagement
- 0.4-0.6: Stable or slightly changing engagement  
- 0.7-1.0: Increasing engagement

**objection_count** (0.0-1.0): Level of customer objections/resistance
- 0.0: No objections
- 0.5: Moderate objections (price, features, timing)
- 1.0: Strong objections (not interested, won't work, too expensive)

**value_proposition_mentions** (0.0-1.0): How well sales rep articulated value
- 0.0: No value mentioned
- 0.5: Some benefits mentioned
- 1.0: Strong, clear value propositions presented

**technical_depth** (0.0-1.0): How technical/detailed the conversation is
- 0.0: Basic, non-technical discussion
- 0.5: Some technical terms or concepts
- 1.0: Highly technical, detailed technical discussion

**urgency_level** (0.0-1.0): Time pressure or urgency indicators
- 0.0: No urgency mentioned
- 0.5: Some time considerations
- 1.0: High urgency, deadlines, time-sensitive

**competitive_context** (0.0-1.0): Mentions of competitors or alternatives
- 0.0: No competitive mentions
- 0.5: Some comparison or alternatives discussed
- 1.0: Heavy competitive comparison focus

**pricing_sensitivity** (0.0-1.0): Customer's focus on cost/budget
- 0.0: Price not discussed or not a concern
- 0.5: Some price discussion or budget considerations
- 1.0: Heavy focus on price, budget constraints, cost concerns

**decision_authority_signals** (0.0-1.0): Customer's decision-making power
- 0.0-0.3: Low authority ("need to check with boss", "team decision")
- 0.4-0.6: Moderate authority (some input but not final decision)
- 0.7-1.0: High authority ("I decide", "my budget", "I approve")

CRITICAL: Respond with ONLY the JSON object. No explanations or additional text.

JSON Response:"""

        try:
            llm_response = self.llm(
                prompt,
                max_tokens=300,
                temperature=0.1,
                stop=["\n\n", "```"],
            )
            raw_llm_output = llm_response['choices'][0]['text'].strip()
            logger.debug(f"DEBUG: LLM Raw Output for comprehensive metrics: '{raw_llm_output}'")

            # Parse the JSON
            json_match = re.search(r"\{.*\}", raw_llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    
                    # Validate and normalize the JSON response
                    validated_metrics = self._validate_and_normalize_metrics(parsed_json)
                    logger.info(f"Successfully parsed comprehensive LLM metrics with {len(validated_metrics)} fields")
                    return validated_metrics
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON from LLM output: '{json_str}'. Error: {e}")
            else:
                logger.warning(f"No JSON object found in LLM output: '{raw_llm_output}'")
        
        except Exception as e:
            logger.error(f"LLM comprehensive metrics analysis failed: {e}", exc_info=True)
        
        # Fallback to intelligent defaults
        logger.warning("Using intelligent fallback metrics due to LLM parsing failure")
        return self._get_fallback_metrics(history, turn_number)

    def _validate_and_normalize_metrics(self, parsed_json: Dict) -> Dict:
        """Validate and normalize the LLM-provided metrics."""
        validated = {}
        
        # Numeric fields (0.0-1.0)
        numeric_fields = [
            'customer_engagement', 'sales_effectiveness', 'engagement_trend',
            'objection_count', 'value_proposition_mentions', 'technical_depth',
            'urgency_level', 'competitive_context', 'pricing_sensitivity',
            'decision_authority_signals'
        ]
        
        for field in numeric_fields:
            value = parsed_json.get(field, 0.5)
            if isinstance(value, (int, float)):
                validated[field] = float(np.clip(value, 0.0, 1.0))
            else:
                validated[field] = 0.5
        
        # String fields with valid options
        style_options = [
            "casual_friendly", "direct_professional", "technical_detailed", "consultative_advisory",
            "empathetic_supportive", "skeptical_challenging", "urgent_time_pressed", "confused_overwhelmed",
            "knowledgeable_assertive", "storytelling_narrative"
        ]
        validated['conversation_style'] = parsed_json.get('conversation_style', 'direct_professional')
        if validated['conversation_style'] not in style_options:
            validated['conversation_style'] = 'direct_professional'
        
        flow_options = [
            "standard_linear", "multiple_objection_loops", "subject_switching", "interrupted_followup",
            "technical_deep_dive", "competitive_comparison", "gradual_discovery", "immediate_interest",
            "initial_rejection", "stakeholder_expansion", "pricing_negotiation", "implementation_concerns",
            "value_justification", "relationship_building", "multi_session", "demo_walkthrough"
        ]
        validated['conversation_flow'] = parsed_json.get('conversation_flow', 'standard_linear')
        if validated['conversation_flow'] not in flow_options:
            validated['conversation_flow'] = 'standard_linear'
        
        channel_options = ["email", "live_chat", "phone_call", "video_call", "in_person", "sms", "social_media"]
        validated['communication_channel'] = parsed_json.get('communication_channel', 'email')
        if validated['communication_channel'] not in channel_options:
            validated['communication_channel'] = 'email'
        
        # Array field for customer needs
        need_options = [
            "efficiency", "cost_reduction", "growth", "compliance", "integration",
            "usability", "reliability", "security", "support", "analytics"
        ]
        needs = parsed_json.get('primary_customer_needs', ['efficiency', 'cost_reduction'])
        if isinstance(needs, list):
            validated['primary_customer_needs'] = [need for need in needs if need in need_options][:3]
        else:
            validated['primary_customer_needs'] = ['efficiency', 'cost_reduction']
        
        if not validated['primary_customer_needs']:
            validated['primary_customer_needs'] = ['efficiency', 'cost_reduction']
        
        return validated

    def _get_fallback_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict:
        """Generate intelligent fallback metrics when LLM is not available."""
        conversation_length = len(history)
        
        # Basic text analysis for fallbacks
        customer_text = " ".join([msg['message'].lower() for msg in history if msg['speaker'] == 'customer'])
        sales_text = " ".join([msg['message'].lower() for msg in history if msg['speaker'] == 'sales_rep'])
        
        # Intelligent fallback based on content analysis
        engagement = 0.5
        effectiveness = 0.5
        
        # Adjust based on obvious signals
        if any(signal in customer_text for signal in ['buy', 'purchase', 'interested', 'yes', 'sounds good']):
            engagement = min(0.8, engagement + 0.3)
        
        if any(obj in customer_text for obj in ['expensive', 'costly', 'not interested', 'no']):
            engagement = max(0.2, engagement - 0.3)
        
        if any(poor in sales_text for poor in ['ok', 'sure', 'fine']) and len(sales_text) < 20:
            effectiveness = max(0.2, effectiveness - 0.3)
        
        return {
            'customer_engagement': engagement,
            'sales_effectiveness': effectiveness,
            'conversation_style': 'direct_professional',
            'conversation_flow': 'standard_linear',
            'communication_channel': 'email',
            'primary_customer_needs': ['efficiency', 'cost_reduction'],
            'engagement_trend': 0.5,
            'objection_count': 0.3 if any(obj in customer_text for obj in ['expensive', 'costly']) else 0.1,
            'value_proposition_mentions': 0.4,
            'technical_depth': 0.3,
            'urgency_level': 0.2,
            'competitive_context': 0.1,
            'pricing_sensitivity': 0.4 if 'price' in customer_text or 'cost' in customer_text else 0.2,
            'decision_authority_signals': 0.5
        }

    def _generate_probability_trajectory(self, history: List[Dict[str, str]], comprehensive_metrics: Dict) -> Dict[int, float]:
        """Generate realistic probability trajectory using comprehensive LLM metrics."""
        trajectory = {}
        num_turns = len(history)
        
        if num_turns == 0:
            return {0: 0.5}
        
        # Use comprehensive metrics for trajectory generation
        engagement = comprehensive_metrics.get('customer_engagement', 0.5)
        effectiveness = comprehensive_metrics.get('sales_effectiveness', 0.5)
        engagement_trend = comprehensive_metrics.get('engagement_trend', 0.5)
        objection_count = comprehensive_metrics.get('objection_count', 0.3)
        
        # More sophisticated trajectory calculation
        base_prob = 0.2
        trend_factor = (engagement * 0.5 + effectiveness * 0.3 + engagement_trend * 0.2)
        
        # Adjust for objections
        objection_penalty = objection_count * 0.3
        
        for i in range(num_turns):
            progress = i / max(1, num_turns - 1)
            
            # Calculate probability with sophisticated weighting
            prob = base_prob + (trend_factor - objection_penalty - 0.4) * progress * 1.2
            
            # Apply engagement trend
            if engagement_trend > 0.7:
                prob += progress * 0.2  # Increasing engagement boosts trajectory
            elif engagement_trend < 0.3:
                prob -= progress * 0.15  # Decreasing engagement hurts trajectory
            
            # Smooth progression with some realistic variation
            if i > 0:
                prev_prob = trajectory[i-1]
                max_change = 0.2
                prob = max(prev_prob - max_change, min(prev_prob + max_change, prob))
            
            # Add small realistic noise
            noise = random.uniform(-0.02, 0.02)
            prob = np.clip(prob + noise, 0.05, 0.95)
            
            trajectory[i] = round(prob, 4)
        
        return trajectory

    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, float]:
        conversation_length = float(len(history))
        progress_metric = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        
        # Get comprehensive metrics from LLM
        comprehensive_metrics = self._get_comprehensive_metrics_from_llm(history, turn_number)
        
        # Generate sophisticated probability trajectory
        probability_trajectory = self._generate_probability_trajectory(history, comprehensive_metrics)
        
        # Build final metrics dictionary combining all sources
        metrics = {
            # Core metrics (required by PPO model)
            'customer_engagement': comprehensive_metrics['customer_engagement'],
            'sales_effectiveness': comprehensive_metrics['sales_effectiveness'],
            'conversation_length': conversation_length,
            'outcome': 0.5,  # Standard placeholder for inference
            'progress': progress_metric,
            
            # Enhanced conversation characteristics
            'conversation_style': comprehensive_metrics['conversation_style'],
            'conversation_flow': comprehensive_metrics['conversation_flow'],
            'communication_channel': comprehensive_metrics['communication_channel'],
            'primary_customer_needs': comprehensive_metrics['primary_customer_needs'],
            'probability_trajectory': probability_trajectory,
            
            # Sophisticated behavioral analytics (all LLM-derived)
            'engagement_trend': comprehensive_metrics['engagement_trend'],
            'objection_count': comprehensive_metrics['objection_count'],
            'value_proposition_mentions': comprehensive_metrics['value_proposition_mentions'],
            'technical_depth': comprehensive_metrics['technical_depth'],
            'urgency_level': comprehensive_metrics['urgency_level'],
            'competitive_context': comprehensive_metrics['competitive_context'],
            'pricing_sensitivity': comprehensive_metrics['pricing_sensitivity'],
            'decision_authority_signals': comprehensive_metrics['decision_authority_signals']
        }
        
        llm_derived = self.llm is not None
        logger.info(f"Comprehensive Metrics Analysis (LLM derived: {llm_derived}) - "
                   f"Engagement: {metrics['customer_engagement']:.2f}, "
                   f"Effectiveness: {metrics['sales_effectiveness']:.2f}, "
                   f"Style: {metrics['conversation_style']}, "
                   f"Flow: {metrics['conversation_flow']}")
        
        return metrics

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
            logger.debug(f"LLM prompt messages for response generation: {messages_for_llm}")
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
    """Azure OpenAI embedding provider with basic enhanced metrics."""

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
        self.deployment_name = deployment 
        self.expected_dim = expected_dim
        self.native_dim = 0 
        self.MAX_TURNS_REFERENCE = 20

        try:
            logger.info(f"Testing Azure OpenAI connection with embedding deployment: {self.deployment_name}")
            test_response = self.client.embeddings.create(
                input="test",
                model=self.deployment_name
            )
            self.native_dim = len(test_response.data[0].embedding)
            logger.info(f"Azure embeddings initialized. Native dim: {self.native_dim}, Expected dim: {self.expected_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure embeddings (model: {self.deployment_name}): {e}")
            raise

    def get_embedding(self, text: str, turn_number: int) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment_name
            )
            embedding_native = np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Azure embedding API call failed: {e}")
            dim_to_use = self.native_dim if self.native_dim > 0 else self.expected_dim
            embedding_native = np.zeros(dim_to_use, dtype=np.float32)

        if self.native_dim == 0 and embedding_native.shape[0] > 0 : 
             self.native_dim = embedding_native.shape[0]

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

    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, float]:
        logger.info("AzureEmbeddings using basic enhanced metrics. For full LLM analysis, use OpenSource backend with LLM.")
        
        conversation_length = float(len(history))
        progress_metric = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        
        # Basic content analysis for Azure backend
        customer_text = " ".join([msg['message'].lower() for msg in history if msg['speaker'] == 'customer'])
        
        engagement = 0.5
        effectiveness = 0.5
        
        # Basic signal detection
        if any(signal in customer_text for signal in ['buy', 'purchase', 'interested']):
            engagement = 0.7
        if any(obj in customer_text for obj in ['expensive', 'costly', 'not interested']):
            engagement = 0.3
        
        return {
            'customer_engagement': engagement,
            'sales_effectiveness': effectiveness,
            'conversation_length': conversation_length,
            'outcome': 0.5, 
            'progress': progress_metric,
            
            # Basic versions of enhanced metrics
            'conversation_style': 'direct_professional',
            'conversation_flow': 'standard_linear',
            'communication_channel': 'email',
            'primary_customer_needs': ['efficiency', 'cost_reduction'],
            'probability_trajectory': {i: 0.5 for i in range(len(history))},
            
            # Default derived metrics
            'engagement_trend': 0.5,
            'objection_count': 0.3 if any(obj in customer_text for obj in ['expensive', 'costly']) else 0.1,
            'value_proposition_mentions': 0.3,
            'technical_depth': 0.4,
            'urgency_level': 0.2,
            'competitive_context': 0.1,
            'pricing_sensitivity': 0.4 if 'price' in customer_text or 'cost' in customer_text else 0.2,
            'decision_authority_signals': 0.5
        }

    def generate_response(
        self,
        history: List[Dict[str, str]],
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> str:
        logger.warning(
            "AzureEmbeddings.generate_response is a placeholder. "
            "For actual LLM responses with Azure backend, integrate with Azure OpenAI Chat Completions."
        )
        return "Thank you for your inquiry. An agent will follow up with you."