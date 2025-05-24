# DeepMost - Sales Conversion Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python package for predicting sales conversion probability using reinforcement learning. Get accurate conversion predictions with just 3 lines of code! DeepMost leverages advanced PPO-based models and uses GGUF-based LLMs for dynamic conversation analysis via JSON, providing richer and more reliable insights.

## 🚀 Features

- **Simple API**: Predict conversion probability with just 3 lines of code.
- **Reinforcement Learning**: Advanced PPO-based model trained on real sales conversations.
- **Dynamic Metrics with GGUF LLMs**: Uses GGUF models (via `llama-cpp-python`) to generate `customer_engagement` and `sales_effectiveness` scores by prompting for **JSON output**. This is crucial for accurate predictions if the PPO model was trained with such dynamic metrics.
- **Open-Source Backend**: Leverages HuggingFace sentence-transformers for embeddings (e.g., `BAAI/bge-m3` providing 1024-dim embeddings) and GGUF models for metrics/responses.
- **Intelligent Sales Responses**: LLM integration for generating sales representative replies.
- **Auto-Download**: Reinforcement learning PPO model and GGUF LLM (if specified via HuggingFace repo) download automatically on first use and are cached.
- **GPU Acceleration**: Full CUDA support for faster PPO inference and LLM operations.
- **Conversation Analysis**: Detailed metrics including dynamic engagement and effectiveness scores extracted via JSON.

## 📦 Installation

Requires **Python 3.11, does not support on other versions**.

### Basic Installation
Installs the core package with CPU-based open-source embeddings. For dynamic metrics (recommended for accuracy) and response generation, install with `[gpu]` and provide an `llm_model`.
```bash
pip install deepmost
```

### With GPU Support (for local GGUF LLMs & GPU-accelerated Embeddings)
Enables GPU acceleration for open-source embeddings and local GGUF LLMs. **This is recommended for using the LLM-driven dynamic metrics feature.**
```bash
pip install deepmost[gpu]
```


## 🎯 Quick Start

### Simplest Usage (3 lines with LLM for Dynamic Metrics)
This uses the open-source backend. Providing an `llm_model` is **highly recommended** for accurate dynamic metrics (engagement, effectiveness) as the default PPO model is trained with them.
```python
from deepmost import sales

conversation = ["Hi, I need a CRM for my business", "I'd be happy to help! What's your team size?"]

# For optimal accuracy, pass an llm_model to enable dynamic metrics
probability = sales.predict(
    conversation,
    llm_model="unsloth/Llama-3.2-3B-Instruct-GGUF" # Recommended GGUF model
)
# The above will initialize a temporary Agent with the LLM.
# For multiple predictions, initialize an Agent instance once (see below).

print(f"Conversion probability (with LLM-derived metrics): {probability:.1%}")
```

### Using the Agent API (Recommended for Control & Efficiency)
```python
from deepmost import sales

# Initialize agent. Provide an llm_model for dynamic, LLM-derived metrics.
# Default PPO model expects these.
# Using BAAI/bge-m3 for embeddings (default, 1024 dimensions)
agent = sales.Agent(llm_model="unsloth/Llama-3.2-3B-Instruct-GGUF")

conversation = [
    {"speaker": "customer", "message": "I'm looking for a CRM solution for about 50 users."},
    {"speaker": "sales_rep", "message": "Excellent! Our Enterprise plan is perfect for teams of that size and offers advanced features. What are your key requirements?"},
    {"speaker": "customer", "message": "We need robust reporting, Salesforce integration, and an easy-to-use interface."},
    {"speaker": "sales_rep", "message": "The Enterprise plan has all of that, including customizable dashboards and a dedicated onboarding specialist. Would you like to schedule a demo?"}
]

result = agent.predict(conversation)
print(f"Probability: {result['probability']:.1%}")
print(f"Status: {result['status']}")
print(f"Metrics (LLM-derived): {result['metrics']}")
```

## 🤖 Generate Sales Responses
This feature requires an `llm_model` to be configured.
```python
from deepmost import sales

# Initialize with LLM support (GGUF model path or HuggingFace GGUF repo ID)
agent = sales.Agent(llm_model="unsloth/Llama-3.2-3B-Instruct-GGUF")

conversation_history = [
    {"speaker": "customer", "message": "I'm interested in your CRM"},
    {"speaker": "sales_rep", "message": "Great! What are your main requirements?"}
]

result = agent.predict_with_response(
    conversation=conversation_history,
    user_input="We need something that integrates with Hubspot and has good reporting.",
    system_prompt="You are a helpful sales representative for AcmeCRM."
)

print(f"Generated Response: {result['response']}")
print(f"Conversion Probability: {result['prediction']['probability']:.1%}")
print(f"Suggested Action: {result['prediction']['suggested_action']}")
```

## 🔧 Configuration Options

### Using a Custom PPO Model
Your custom PPO model should be trained with an observation space compatible with DeepMost's state vector structure (embedding + 5 metrics + turn info + probability history).
```python
from deepmost import sales

agent = sales.Agent(
    model_path="/path/to/your/ppo_model.zip", # Your trained PPO model
    embedding_model="BAAI/bge-m3", # Ensure this matches your PPO model's expected embedding source
    llm_model="unsloth/Llama-3.2-3B-Instruct-GGUF" # If your custom PPO model also uses LLM-derived metrics
)
```

### Advanced Configuration
```python
from deepmost import sales

agent = sales.Agent(
    # PPO Model settings
    model_path="/path/to/custom/ppo_model.zip", 
    auto_download=True,  # Auto-download default PPO model if model_path is None or file not found

    # Embedding settings
    embedding_model="BAAI/bge-m3",  # Default: HuggingFace sentence-transformer model (1024-dim)
                                   # PPO model expects embeddings of this type/dimension.

    # LLM settings (for dynamic metrics via JSON & response generation)
    # Provide a GGUF model path or a HuggingFace repo ID for a GGUF model.
    # Highly recommended for accuracy with the default PPO model.
    llm_model="unsloth/Llama-3.2-3B-Instruct-GGUF", # Recommended HuggingFace GGUF repo
    # OR
    # llm_model="/path/to/local/model.gguf",

    # Performance
    use_gpu=True  # Enable GPU acceleration for PPO, embeddings, and LLM
)
```

## 📊 Understanding the Output
The `predict` method returns a dictionary:
```python
result = agent.predict(conversation)

# result contains:
{
    'probability': 0.752,  # Conversion probability (0.0 to 1.0)
    'turn': 3,            # Current conversation turn number (0-indexed)
    'status': '🟢 High',  # Status indicator based on probability
    'metrics': {          # Metrics used for this prediction step
        'customer_engagement': 0.8, # Dynamically generated by LLM (via JSON) if llm_model is configured
        'sales_effectiveness': 0.7, # Dynamically generated by LLM (via JSON) if llm_model is configured
        'conversation_length': 4.0, # Number of messages in the conversation history
        'outcome': 0.5,             # Placeholder for PPO model (always 0.5 at inference)
        'progress': 0.15            # Normalized progress (current_turn / max_turns_reference)
    },
    'suggested_action': "Focus on closing: Propose next steps..." # Actionable suggestion
}
```

### Status Indicators
- 🟢 **High** (Probability ≥ 70%): Conversion highly likely.
- 🟡 **Medium** (Probability ≥ 50%): Good potential.
- 🟠 **Low** (Probability ≥ 30%): Needs improvement.
- 🔴 **Very Low** (Probability < 30%): Unlikely to convert.

## 🗂️ Conversation Format
The package accepts conversations in multiple formats for the `conversation` argument:

### Simple List Format (Alternating Customer/Sales Rep)
```python
conversation = [
    "Customer message 1", # Assumed customer
    "Sales rep response 1", # Assumed sales_rep
    # ...
]
```

### Structured Format (Recommended)
```python
conversation = [
    {"speaker": "customer", "message": "I need help"},
    {"speaker": "sales_rep", "message": "I'm here to help!"}
]
```
Supported speaker values: 'customer', 'user'; 'sales_rep', 'assistant', 'agent', 'bot', 'model'.

### Alternative Keys (OpenAI Style)
```python
conversation = [
    {"role": "user", "content": "I need a CRM"},
    {"role": "assistant", "content": "Let me help you find the right solution"}
]
```

## 🛠️ Troubleshooting

### GPU Support & `llama-cpp-python`
- Ensure compatible NVIDIA GPU, CUDA toolkit, and drivers.
- If `deepmost[gpu]` installation fails, try installing `llama-cpp-python` manually first, potentially with `CMAKE_ARGS` for CUDA: `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir`.
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available(): print(f"CUDA Version: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
```

### PPO Model Download Issues
If automatic download of the PPO model fails:
```python
from deepmost.core.utils import download_model
from deepmost.sales import _get_default_model_info # Ensure this function still works without Azure logic or defaults appropriately

# For open-source default
model_url, default_model_path = _get_default_model_info(use_azure=False) # Explicitly open-source
print(f"PPO model URL: {model_url}, Default path: {default_model_path}")
# To manually download:
# import os
# os.makedirs(os.path.dirname(default_model_path), exist_ok=True)
# download_model(model_url, default_model_path)
# agent = sales.Agent(model_path=default_model_path, auto_download=False, ...)
```

### LLM Model Issues (GGUF)
- **Download & Cache**: `Llama.from_pretrained` (used for HuggingFace repo IDs) caches models typically in `~/.cache/huggingface/hub`. Ensure connectivity and disk space.
- **Compatibility**: Use GGUF models compatible with your `llama-cpp-python` version.
- **Memory**: Large LLMs need significant RAM/VRAM. For systems with limited resources, consider smaller models:
    ```python
    agent = sales.Agent(
        llm_model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", # Example smaller model
        use_gpu=True
    )
    ```
- **Metrics Fallback**: If an LLM is specified but fails to load or provide valid JSON metrics, DeepMost logs a warning and uses default static metrics (0.5 for engagement/effectiveness), impacting prediction accuracy. Check logs for `DEBUG: LLM Raw Output for JSON metrics:` to see the LLM's direct response.

## 📈 Performance Tips

1.  **Persistent Agent**: Initialize `sales.Agent()` once and reuse it.
2.  **GPU Optimization**: Ensure `use_gpu=True`. Keep drivers/CUDA updated.
3.  **Appropriate LLM Size**: Balance LLM quality with resources. Models like `unsloth/Llama-3.2-3B-Instruct-GGUF` or other 1B-3B parameter GGUF models (e.g., Phi-3 Instruct Mini) often provide a good balance for the metrics task. Larger models (7B+) may offer higher quality responses/metrics at the cost of more resources.

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) (you'll need to create this file).
```bash
git clone https://github.com/DeepMostInnovations/deepmost.git
cd deepmost
pip install -e .[dev]
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- PPO: [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Embeddings: [Sentence Transformers](https://www.sbert.net/) (e.g., `BAAI/bge-m3`)
- GGUF LLM Support: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/DeepMostInnovations/deepmost/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DeepMostInnovations/deepmost/discussions)
- **Email**: support@deepmost.ai

## 🔗 Links

- **PyPI**: [https://pypi.org/project/deepmost/](https://pypi.org/project/deepmost/)
- **GitHub**: [https://github.com/DeepMostInnovations/deepmost](https://github.com/DeepMostInnovations/deepmost)
- **Docs**: [https://deepmost.readthedocs.io/](https://deepmost.readthedocs.io/) (To be set up)
- **Models**: [https://huggingface.co/DeepMostInnovations](https://huggingface.co/DeepMostInnovations)

---

Made with ❤️ by [DeepMost Innovations](https://deepmost.ai)

