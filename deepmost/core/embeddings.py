"""Embedding providers for different backends"""

import numpy as np
import torch
import logging
from typing import List, Dict, Optional, Protocol
from transformers import AutoTokenizer, AutoModel
import re # Still useful for some cleanup or simple cases
import json # For parsing JSON output
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
    """Open-source embedding provider using HuggingFace models and optional LLM for metrics/responses."""

    def __init__(
        self,
        model_name: str, # Embedding model name
        device: torch.device,
        expected_dim: int,
        llm_model: Optional[str] = None  # Path to local GGUF or HF repo ID for GGUF LLM
    ):
        self.device = device
        self.expected_dim = expected_dim
        self.MAX_TURNS_REFERENCE = 20

        logger.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.native_dim = self.model.config.hidden_size
        logger.info(f"Embedding model loaded. Native dim: {self.native_dim}, Expected dim: {self.expected_dim}")

        # Initialize conversation style and flow options (from dataset generator)
        self.conversation_styles = [
            "casual_friendly", "direct_professional", "technical_detailed",
            "consultative_advisory", "empathetic_supportive", "skeptical_challenging",
            "urgent_time_pressed", "confused_overwhelmed", "knowledgeable_assertive",
            "storytelling_narrative"
        ]
        
        self.conversation_flows = [
            "standard_linear", "multiple_objection_loops", "subject_switching",
            "interrupted_followup", "technical_deep_dive", "competitive_comparison",
            "gradual_discovery", "immediate_interest", "initial_rejection",
            "stakeholder_expansion", "pricing_negotiation", "implementation_concerns",
            "value_justification", "relationship_building", "multi_session", "demo_walkthrough"
        ]
        
        self.communication_channels = [
            "email", "live_chat", "phone_call", "video_call", 
            "in_person", "sms", "social_media"
        ]
        
        # Customer needs for sophisticated analysis
        self.customer_needs = [
            {"type": "efficiency", "keywords": ["faster", "streamline", "automate", "time-consuming", "manual", "process", "workflow"]},
            {"type": "cost_reduction", "keywords": ["expenses", "budget", "save money", "affordable", "cost-effective", "ROI", "investment"]},
            {"type": "growth", "keywords": ["scale", "expand", "increase revenue", "market share", "competitive", "opportunity", "growth"]},
            {"type": "compliance", "keywords": ["regulations", "requirements", "standards", "audit", "legal", "compliance", "risk"]},
            {"type": "integration", "keywords": ["connect", "compatible", "ecosystem", "work with", "existing systems", "API", "integration"]},
            {"type": "usability", "keywords": ["easy to use", "intuitive", "learning curve", "training", "user-friendly", "simple", "interface"]},
            {"type": "reliability", "keywords": ["uptime", "stable", "dependable", "trust", "consistent", "failover", "backup"]},
            {"type": "security", "keywords": ["protect", "data security", "encryption", "sensitive information", "breach", "privacy", "secure"]},
            {"type": "support", "keywords": ["help", "customer service", "response time", "training", "documentation", "support team", "assistance"]},
            {"type": "analytics", "keywords": ["insights", "reporting", "dashboard", "metrics", "data", "visibility", "analytics"]}
        ]

        self.llm = None
        if llm_model:
            logger.info(f"Attempting to load GGUF LLM: {llm_model}")
            try:
                from llama_cpp import Llama
                
                llama_params = {
                    "n_gpu_layers": 30,
                    "n_ctx": 2048, # Model context window
                    "verbose": False 
                }

                if "/" in llm_model and not llm_model.lower().endswith(".gguf"):
                    repo_id = llm_model
                    logger.info(f"LLM '{repo_id}' is a HuggingFace repo. Using Llama.from_pretrained.")
                    
                    # Try with a common pattern or None to let LlamaCPP pick
                    gguf_filename_pattern = "*Q4_K_M.gguf" # Example, could be None
                    
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
                self.llm = None # Ensure self.llm is None if loading fails
            except Exception as e:
                logger.warning(f"Failed to load GGUF LLM '{llm_model}': {e}. LLM features may be unavailable.")
                self.llm = None # Ensure self.llm is None
        
        if not self.llm:
            logger.info(
                "No LLM loaded. Metric analysis will use defaults, and response generation will be basic. "
                "This may significantly affect prediction accuracy if PPO model was trained with LLM-derived metrics."
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

    def _detect_conversation_style(self, history: List[Dict[str, str]]) -> str:
        """Detect the conversation style based on message patterns."""
        if not history:
            return random.choice(self.conversation_styles)
        
        # Analyze conversation characteristics
        message_lengths = [len(msg['message'].split()) for msg in history]
        avg_length = np.mean(message_lengths) if message_lengths else 10
        
        # Check for technical terms
        tech_keywords = ['API', 'integration', 'security', 'infrastructure', 'scalability', 'architecture']
        has_tech_terms = any(any(keyword.lower() in msg['message'].lower() for keyword in tech_keywords) for msg in history)
        
        # Check for urgency indicators
        urgency_keywords = ['urgent', 'asap', 'quickly', 'deadline', 'immediate', 'time-sensitive']
        has_urgency = any(any(keyword.lower() in msg['message'].lower() for keyword in urgency_keywords) for msg in history)
        
        # Check for casual language
        casual_indicators = ['hey', 'yeah', 'cool', 'awesome', 'thanks', '!', 'lol', 'btw']
        casual_score = sum(sum(indicator in msg['message'].lower() for indicator in casual_indicators) for msg in history)
        
        # Check for confusion indicators
        confusion_keywords = ['confused', 'not sure', 'unclear', 'help me understand', 'what do you mean']
        has_confusion = any(any(keyword.lower() in msg['message'].lower() for keyword in confusion_keywords) for msg in history)
        
        # Determine style based on characteristics
        if has_tech_terms and avg_length > 15:
            return "technical_detailed"
        elif has_urgency:
            return "urgent_time_pressed"
        elif casual_score > 2:
            return "casual_friendly"
        elif has_confusion:
            return "confused_overwhelmed"
        elif avg_length > 20:
            return "consultative_advisory"
        elif avg_length < 8:
            return "direct_professional"
        else:
            return random.choice(self.conversation_styles)

    def _detect_conversation_flow(self, history: List[Dict[str, str]]) -> str:
        """Detect the conversation flow pattern based on message sequence."""
        if len(history) < 3:
            return "standard_linear"
        
        # Check for multiple objections
        objection_keywords = ['but', 'however', 'concern', 'worry', 'issue', 'problem', 'expensive', 'cost']
        objection_count = sum(sum(keyword.lower() in msg['message'].lower() for keyword in objection_keywords) for msg in history if msg['speaker'] == 'customer')
        
        # Check for topic switching
        topics = ['price', 'feature', 'integration', 'support', 'timeline', 'demo', 'trial']
        topics_mentioned = set()
        for msg in history:
            for topic in topics:
                if topic in msg['message'].lower():
                    topics_mentioned.add(topic)
        
        # Check for competitive mentions
        competitor_keywords = ['competitor', 'alternative', 'vs', 'compare', 'other solution']
        has_competition = any(any(keyword.lower() in msg['message'].lower() for keyword in competitor_keywords) for msg in history)
        
        # Check for immediate interest
        interest_keywords = ['interested', 'sounds good', 'perfect', 'exactly', 'great', 'love it']
        early_interest = any(any(keyword.lower() in history[i]['message'].lower() for keyword in interest_keywords) for i in range(min(3, len(history))) if history[i]['speaker'] == 'customer')
        
        # Check for pricing focus
        pricing_keywords = ['price', 'cost', 'budget', 'expensive', 'affordable', 'discount']
        pricing_focus = sum(sum(keyword.lower() in msg['message'].lower() for keyword in pricing_keywords) for msg in history) >= 3
        
        # Determine flow pattern
        if objection_count >= 3:
            return "multiple_objection_loops"
        elif len(topics_mentioned) >= 4:
            return "subject_switching"
        elif has_competition:
            return "competitive_comparison"
        elif early_interest:
            return "immediate_interest"
        elif pricing_focus:
            return "pricing_negotiation"
        elif any('demo' in msg['message'].lower() for msg in history):
            return "demo_walkthrough"
        else:
            return random.choice(self.conversation_flows)

    def _detect_communication_channel(self, history: List[Dict[str, str]]) -> str:
        """Detect the communication channel based on message characteristics."""
        if not history:
            return random.choice(self.communication_channels)
        
        # Analyze message characteristics
        message_lengths = [len(msg['message']) for msg in history]
        avg_length = np.mean(message_lengths) if message_lengths else 100
        
        # Check for email indicators
        email_indicators = ['subject:', 'dear', 'sincerely', 'best regards', 'email', 'attachment']
        has_email_markers = any(any(indicator.lower() in msg['message'].lower() for indicator in email_indicators) for msg in history)
        
        # Check for chat indicators
        chat_indicators = ['hey', 'hi there', '👍', '😊', 'lol', 'brb', 'u', 'ur']
        has_chat_markers = any(any(indicator in msg['message'].lower() for indicator in chat_indicators) for msg in history)
        
        # Check for phone/video indicators
        verbal_indicators = ['can you hear me', 'video', 'screen share', 'mute', 'audio', 'call']
        has_verbal_markers = any(any(indicator.lower() in msg['message'].lower() for indicator in verbal_indicators) for msg in history)
        
        # Determine channel
        if has_email_markers or avg_length > 200:
            return "email"
        elif has_chat_markers and avg_length < 100:
            return "live_chat"
        elif has_verbal_markers:
            return random.choice(["phone_call", "video_call"])
        elif avg_length < 50:
            return "sms"
        else:
            return random.choice(self.communication_channels)

    def _detect_customer_needs(self, history: List[Dict[str, str]]) -> List[str]:
        """Detect primary customer needs based on conversation content."""
        detected_needs = []
        
        # Combine all customer messages
        customer_text = " ".join([msg['message'].lower() for msg in history if msg['speaker'] == 'customer'])
        
        # Check each need type
        for need in self.customer_needs:
            keyword_matches = sum(1 for keyword in need['keywords'] if keyword in customer_text)
            if keyword_matches >= 2:  # Threshold for detecting a need
                detected_needs.append(need['type'])
        
        # Return top 3 detected needs or random ones if none detected
        if detected_needs:
            return detected_needs[:3]
        else:
            return random.sample([need['type'] for need in self.customer_needs], 3)

    def _generate_probability_trajectory(self, history: List[Dict[str, str]], final_engagement: float, final_effectiveness: float) -> Dict[int, float]:
        """Generate realistic probability trajectory for the conversation."""
        trajectory = {}
        num_turns = len(history)
        
        if num_turns == 0:
            return {0: 0.5}
        
        # Determine overall trend based on engagement and effectiveness
        base_prob = 0.3  # Starting probability
        trend_factor = (final_engagement + final_effectiveness) / 2
        
        # Generate trajectory with realistic variations
        for i in range(num_turns):
            # Base progression
            progress = i / max(1, num_turns - 1)
            
            # Calculate probability with trend and some randomness
            prob = base_prob + (trend_factor - 0.5) * progress
            
            # Add some realistic variation
            if i > 0:
                prev_prob = trajectory[i-1]
                # Limit big jumps
                max_change = 0.15
                prob = max(prev_prob - max_change, min(prev_prob + max_change, prob))
            
            # Add noise but keep reasonable bounds
            noise = random.uniform(-0.05, 0.05)
            prob = np.clip(prob + noise, 0.05, 0.95)
            
            trajectory[i] = round(prob, 4)
        
        return trajectory

    def analyze_metrics(self, history: List[Dict[str, str]], turn_number: int) -> Dict[str, float]:
        conversation_length = float(len(history))
        progress_metric = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        
        customer_engagement = 0.5 # Default
        sales_effectiveness = 0.5 # Default
        llm_derived = False

        if self.llm:
            logger.debug("Analyzing metrics using LLM (attempting JSON output).")
            conversation_text = "\n".join([f"{msg['speaker'].capitalize()}: {msg['message']}" for msg in history])
            
            if not conversation_text.strip():
                logger.warning("Conversation history is empty for LLM metric analysis. Using default metrics.")
            else:
                # Enhanced prompting for more comprehensive analysis
                prompt = f"""Analyze the following sales conversation snippet.
Based ONLY on the provided text, provide scores from 0.0 to 1.0 for "customer_engagement" and "sales_effectiveness".

"customer_engagement" reflects how engaged the customer is:
- 0.0-0.3: Disengaged, short responses, showing little interest
- 0.4-0.6: Moderately engaged, asking some questions, participating
- 0.7-1.0: Highly engaged, asking detailed questions, showing strong interest

"sales_effectiveness" reflects how effective the sales representative's approach is:
- 0.0-0.3: Poor approach, not addressing needs, pushy or unclear
- 0.4-0.6: Adequate approach, some value shown, moderate rapport
- 0.7-1.0: Excellent approach, addressing needs well, building strong value

Conversation:
---
{conversation_text}
---

Your response MUST be a VALID JSON object containing ONLY the keys "customer_engagement" and "sales_effectiveness" with their corresponding float scores. For example:
{{
  "customer_engagement": 0.7,
  "sales_effectiveness": 0.6
}}

JSON Response:
"""
                logger.debug(f"DEBUG: LLM Prompt for JSON metrics:\n{prompt}")
                
                try:
                    llm_response = self.llm(
                        prompt,
                        max_tokens=150,
                        temperature=0.0,
                        stop=["\n\n", "```"],
                    )
                    raw_llm_output = llm_response['choices'][0]['text'].strip()
                    logger.debug(f"DEBUG: LLM Raw Output for JSON metrics: '{raw_llm_output}'")

                    # Attempt to parse the JSON
                    json_match = re.search(r"\{.*\}", raw_llm_output, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            parsed_json = json.loads(json_str)
                            ce = parsed_json.get("customer_engagement")
                            se = parsed_json.get("sales_effectiveness")

                            if isinstance(ce, (float, int)) and isinstance(se, (float, int)):
                                customer_engagement = float(ce)
                                sales_effectiveness = float(se)
                                llm_derived = True
                                logger.debug(f"Successfully parsed JSON metrics: CE={customer_engagement}, SE={sales_effectiveness}")
                            else:
                                logger.warning(f"Parsed JSON but metrics are not valid numbers or missing: {parsed_json}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to decode JSON from LLM output: '{json_str}'. Error: {e}")
                    else:
                        logger.warning(f"No JSON object found in LLM output: '{raw_llm_output}'")
                
                except Exception as e:
                    logger.error(f"LLM metrics analysis (JSON attempt) failed: {e}", exc_info=True)
        
        if not llm_derived:
            logger.warning(
                "LLM not available or failed to derive metrics (customer_engagement, sales_effectiveness) via JSON. "
                "Using default placeholder values (0.5). "
                "If PPO model was trained with LLM-derived metrics, prediction quality will be significantly affected."
            )

        # Generate sophisticated metrics based on conversation analysis
        conversation_style = self._detect_conversation_style(history)
        conversation_flow = self._detect_conversation_flow(history)
        communication_channel = self._detect_communication_channel(history)
        customer_needs = self._detect_customer_needs(history)
        probability_trajectory = self._generate_probability_trajectory(history, customer_engagement, sales_effectiveness)

        metrics = {
            'customer_engagement': np.clip(customer_engagement, 0.0, 1.0),
            'sales_effectiveness': np.clip(sales_effectiveness, 0.0, 1.0),
            'conversation_length': conversation_length,
            'outcome': 0.5, # Standard placeholder for inference
            'progress': progress_metric,
            
            # Enhanced metrics from dataset generator
            'conversation_style': conversation_style,
            'conversation_flow': conversation_flow,
            'communication_channel': communication_channel,
            'primary_customer_needs': customer_needs,
            'probability_trajectory': probability_trajectory,
            
            # Derived metrics
            'engagement_trend': self._calculate_engagement_trend(history),
            'objection_count': self._count_objections(history),
            'value_proposition_mentions': self._count_value_mentions(history),
            'technical_depth': self._calculate_technical_depth(history),
            'urgency_level': self._calculate_urgency_level(history),
            'competitive_context': self._detect_competitive_context(history),
            'pricing_sensitivity': self._calculate_pricing_sensitivity(history),
            'decision_authority_signals': self._detect_decision_authority(history)
        }
        
        logger.info(f"Enhanced Metrics (LLM derived: {llm_derived}): {metrics}")
        return metrics

    def _calculate_engagement_trend(self, history: List[Dict[str, str]]) -> float:
        """Calculate if customer engagement is increasing or decreasing."""
        if len(history) < 4:
            return 0.5
        
        # Compare first half vs second half message lengths and enthusiasm
        mid_point = len(history) // 2
        first_half = history[:mid_point]
        second_half = history[mid_point:]
        
        first_half_lengths = [len(msg['message']) for msg in first_half if msg['speaker'] == 'customer']
        second_half_lengths = [len(msg['message']) for msg in second_half if msg['speaker'] == 'customer']
        
        if not first_half_lengths or not second_half_lengths:
            return 0.5
        
        avg_first = np.mean(first_half_lengths)
        avg_second = np.mean(second_half_lengths)
        
        # Normalize to 0-1 scale
        if avg_first == 0:
            return 0.5
        
        trend = min(1.0, max(0.0, avg_second / avg_first))
        return trend

    def _count_objections(self, history: List[Dict[str, str]]) -> float:
        """Count customer objections in the conversation."""
        objection_keywords = [
            'but', 'however', 'concern', 'worry', 'issue', 'problem', 
            'expensive', 'cost', 'budget', 'not sure', 'hesitant',
            'difficult', 'complex', 'time-consuming', 'risky'
        ]
        
        objection_count = 0
        for msg in history:
            if msg['speaker'] == 'customer':
                text = msg['message'].lower()
                objection_count += sum(1 for keyword in objection_keywords if keyword in text)
        
        # Normalize to reasonable range (0-1)
        return min(1.0, objection_count / 5.0)

    def _count_value_mentions(self, history: List[Dict[str, str]]) -> float:
        """Count mentions of value propositions by sales rep."""
        value_keywords = [
            'save', 'increase', 'improve', 'reduce', 'optimize', 'streamline',
            'efficiency', 'productivity', 'ROI', 'return on investment',
            'benefit', 'advantage', 'value', 'solution'
        ]
        
        value_count = 0
        for msg in history:
            if msg['speaker'] == 'sales_rep':
                text = msg['message'].lower()
                value_count += sum(1 for keyword in value_keywords if keyword in text)
        
        # Normalize to reasonable range (0-1)
        return min(1.0, value_count / 5.0)

    def _calculate_technical_depth(self, history: List[Dict[str, str]]) -> float:
        """Calculate the technical depth of the conversation."""
        technical_keywords = [
            'API', 'integration', 'security', 'infrastructure', 'scalability',
            'architecture', 'database', 'cloud', 'server', 'protocol',
            'encryption', 'authentication', 'deployment', 'configuration'
        ]
        
        tech_count = 0
        total_words = 0
        
        for msg in history:
            text = msg['message'].lower()
            words = text.split()
            total_words += len(words)
            tech_count += sum(1 for word in words if word in [k.lower() for k in technical_keywords])
        
        if total_words == 0:
            return 0.0
        
        return min(1.0, tech_count / (total_words / 100))  # Normalize

    def _calculate_urgency_level(self, history: List[Dict[str, str]]) -> float:
        """Calculate urgency indicators in the conversation."""
        urgency_keywords = [
            'urgent', 'asap', 'quickly', 'deadline', 'immediate', 'rush',
            'time-sensitive', 'soon', 'emergency', 'critical', 'priority'
        ]
        
        urgency_count = 0
        for msg in history:
            text = msg['message'].lower()
            urgency_count += sum(1 for keyword in urgency_keywords if keyword in text)
        
        return min(1.0, urgency_count / 3.0)

    def _detect_competitive_context(self, history: List[Dict[str, str]]) -> float:
        """Detect mentions of competitors or alternatives."""
        competitive_keywords = [
            'competitor', 'alternative', 'vs', 'compare', 'comparison',
            'other solution', 'different option', 'similar product',
            'evaluating', 'vendor', 'proposal'
        ]
        
        competitive_mentions = 0
        for msg in history:
            text = msg['message'].lower()
            competitive_mentions += sum(1 for keyword in competitive_keywords if keyword in text)
        
        return min(1.0, competitive_mentions / 2.0)

    def _calculate_pricing_sensitivity(self, history: List[Dict[str, str]]) -> float:
        """Calculate customer's price sensitivity."""
        pricing_keywords = [
            'price', 'cost', 'budget', 'expensive', 'affordable', 'cheap',
            'discount', 'deal', 'pricing', 'money', 'investment', 'spend'
        ]
        
        pricing_mentions = 0
        for msg in history:
            if msg['speaker'] == 'customer':
                text = msg['message'].lower()
                pricing_mentions += sum(1 for keyword in pricing_keywords if keyword in text)
        
        return min(1.0, pricing_mentions / 3.0)

    def _detect_decision_authority(self, history: List[Dict[str, str]]) -> float:
        """Detect customer's decision-making authority."""
        authority_keywords = [
            'I decide', 'my decision', 'I approve', 'I choose', 'my budget',
            'I have authority', 'final say', 'up to me'
        ]
        
        delegation_keywords = [
            'need to check', 'ask my boss', 'team decision', 'committee',
            'approval needed', 'not my call', 'someone else decides'
        ]
        
        authority_score = 0
        for msg in history:
            if msg['speaker'] == 'customer':
                text = msg['message'].lower()
                authority_score += sum(1 for keyword in authority_keywords if keyword in text)
                authority_score -= sum(1 for keyword in delegation_keywords if keyword in text)
        
        # Normalize to 0-1 range (0.5 = neutral)
        return np.clip(0.5 + authority_score / 4.0, 0.0, 1.0)

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
        logger.warning(
            "AzureEmbeddings is using basic default metrics. "
            "Enhanced metrics require LLM integration for full dataset compatibility."
        )
        conversation_length = float(len(history))
        progress_metric = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        
        # Basic enhanced metrics for Azure backend
        return {
            'customer_engagement': 0.5,
            'sales_effectiveness': 0.5,
            'conversation_length': conversation_length,
            'outcome': 0.5, 
            'progress': progress_metric,
            
            # Basic versions of enhanced metrics
            'conversation_style': random.choice(["direct_professional", "casual_friendly", "technical_detailed"]),
            'conversation_flow': random.choice(["standard_linear", "multiple_objection_loops", "pricing_negotiation"]),
            'communication_channel': random.choice(["email", "phone_call", "live_chat"]),
            'primary_customer_needs': random.sample(["efficiency", "cost_reduction", "growth"], 2),
            'probability_trajectory': {i: 0.5 for i in range(len(history))},
            
            # Default derived metrics
            'engagement_trend': 0.5,
            'objection_count': 0.2,
            'value_proposition_mentions': 0.3,
            'technical_depth': 0.4,
            'urgency_level': 0.2,
            'competitive_context': 0.1,
            'pricing_sensitivity': 0.3,
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