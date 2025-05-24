"""Embedding providers for different backends"""

import numpy as np
import torch
import logging
from typing import List, Dict, Optional, Protocol
from transformers import AutoTokenizer, AutoModel
import re # Still useful for some cleanup or simple cases
import json # For parsing JSON output
import os

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

        self.llm = None
        if llm_model:
            logger.info(f"Attempting to load GGUF LLM: {llm_model}")
            try:
                from llama_cpp import Llama
                
                llama_params = {
                    "n_gpu_layers": -1 if device.type == 'cuda' else 0,
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
                # Prompting for JSON output
                prompt = f"""Analyze the following sales conversation snippet.
Based ONLY on the provided text, provide scores from 0.0 to 1.0 for "customer_engagement" and "sales_effectiveness".
"customer_engagement" reflects how engaged the customer is (0.0 for disengaged, 1.0 for very engaged).
"sales_effectiveness" reflects how effective the sales representative's approach is (0.0 for ineffective, 1.0 for very effective).

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
                logger.info(f"DEBUG: LLM Prompt for JSON metrics:\n{prompt}")
                
                try:
                    # For JSON output, it's often better to use create_chat_completion
                    # if the model supports it well and you can set response_format.
                    # However, with the basic Llama.__call__, we rely on strong prompting.
                    # For more robust JSON, consider using llama_cpp. चलिए (LlamaGrammar)
                    # or setting response_format if using create_chat_completion.
                    
                    # Basic completion call:
                    llm_response = self.llm(
                        prompt,
                        max_tokens=150, # Allow enough tokens for JSON structure and values
                        temperature=0.0, # Very low temperature for structured output
                        stop=["\n\n", "```"], # Stop if it starts writing more text after JSON
                        # Consider adding "}" as a stop token if it tends to over-generate after JSON
                    )
                    raw_llm_output = llm_response['choices'][0]['text'].strip()
                    logger.info(f"DEBUG: LLM Raw Output for JSON metrics: '{raw_llm_output}'")

                    # Attempt to parse the JSON
                    # The LLM might sometimes include ```json ... ``` or other text. Try to extract.
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
                                logger.info(f"Successfully parsed JSON metrics: CE={customer_engagement}, SE={sales_effectiveness}")
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

        metrics = {
            'customer_engagement': np.clip(customer_engagement, 0.0, 1.0),
            'sales_effectiveness': np.clip(sales_effectiveness, 0.0, 1.0),
            'conversation_length': conversation_length,
            'outcome': 0.5, # Standard placeholder for inference
            'progress': progress_metric
        }
        logger.info(f"Final Metrics (LLM derived: {llm_derived}): {metrics}")
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
            # Using create_chat_completion is generally better for conversational responses
            # and some models might handle roles/system prompts better this way.
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
            "AzureEmbeddings is using basic default metrics (customer_engagement: 0.5, sales_effectiveness: 0.5, outcome: 0.5). "
            "If PPO model was trained with LLM-derived metrics, prediction quality will be significantly affected."
        )
        conversation_length = float(len(history))
        progress_metric = min(1.0, turn_number / self.MAX_TURNS_REFERENCE)
        return {
            'customer_engagement': 0.5,
            'sales_effectiveness': 0.5,
            'conversation_length': conversation_length,
            'outcome': 0.5, 
            'progress': progress_metric
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