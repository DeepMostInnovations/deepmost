# DeepMost - Sales Conversion Prediction

[![PyPI version](https://badge.fury.io/py/deepmost.svg)](https://badge.fury.io/py/deepmost)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python package for predicting sales conversion probability using reinforcement learning. Get accurate conversion predictions with just 3 lines of code!

## 🚀 Features

- **Simple API**: Predict conversion probability with just 3 lines of code
- **Reinforcement Learning**: Advanced PPO-based model trained on real sales conversations
- **Dual Backend Support**: Use open-source models or Azure OpenAI embeddings
- **LLM Integration**: Generate intelligent sales responses (optional)
- **Auto-Download**: Model downloads automatically on first use
- **GPU Acceleration**: Full CUDA support for faster inference
- **Conversation Analysis**: Detailed metrics including engagement and effectiveness scores

## 📦 Installation

### Basic Installation
```bash
pip install deepmost
```

### With GPU Support (for local LLMs)
```bash
pip install deepmost[gpu]
```

### With Azure OpenAI Support
```bash
pip install deepmost[azure]
```

### Full Installation (all features)
```bash
pip install deepmost[gpu,azure]
```

## 🎯 Quick Start

### Simplest Usage (3 lines)

```python
from deepmost import sales

# Predict conversion probability
conversation = ["Hi, I need a CRM for my business", "I'd be happy to help! What's your team size?"]
probability = sales.predict(conversation)
print(f"Conversion probability: {probability:.1%}")  # Output: "Conversion probability: 73.5%"
```

### Using the Agent API

```python
from deepmost import sales

# Initialize agent (auto-downloads model on first use)
agent = sales.Agent()

# Predict with structured conversation
conversation = [
    {"speaker": "customer", "message": "I'm looking for a CRM solution"},
    {"speaker": "sales_rep", "message": "I'd be happy to help! What size is your team?"},
    {"speaker": "customer", "message": "We have about 50 sales people"},
    {"speaker": "sales_rep", "message": "Perfect! Our Enterprise plan would be ideal for you."}
]

result = agent.predict(conversation)
print(f"Probability: {result['probability']:.1%}")
print(f"Status: {result['status']}")
print(f"Metrics: {result['metrics']}")
```

## 🤖 Generate Sales Responses

```python
from deepmost import sales

# Initialize with LLM support
agent = sales.Agent(llm_model="unsloth/Llama-3.2-3B-Instruct-GGUF")

# Generate response and predict
conversation_history = [
    {"speaker": "customer", "message": "I'm interested in your CRM"},
    {"speaker": "sales_rep", "message": "Great! What are your main requirements?"}
]

result = agent.predict_with_response(
    conversation=conversation_history,
    user_input="We need something that integrates with Salesforce",
    system_prompt="You are a helpful sales representative for AcmeCRM."
)

print(f"Generated Response: {result['response']}")
print(f"Conversion Probability: {result['prediction']['probability']:.1%}")
print(f"Suggested Action: {result['prediction']['suggested_action']}")
```

## 🔧 Configuration Options

### Using a Custom Model

```python
from deepmost import sales

# Use your own trained model
agent = sales.Agent(model_path="/path/to/your/model.zip")

# Or use the quick predict function
probability = sales.predict(
    conversation=["Hello", "Hi there!"],
    model_path="/path/to/your/model.zip"
)
```

### Using Azure OpenAI Embeddings

```python
from deepmost import sales

agent = sales.Agent(
    azure_api_key="your-api-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="text-embedding-3-large"
)

result = agent.predict(["I need an enterprise solution", "We have great enterprise plans!"])
```

### Advanced Configuration

```python
from deepmost import sales

agent = sales.Agent(
    # Model settings
    model_path="/path/to/custom/model.zip",
    auto_download=False,  # Disable auto-download
    
    # Embedding settings (choose one)
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Open-source
    # OR
    azure_api_key="key",  # Azure OpenAI
    azure_endpoint="endpoint",
    azure_deployment="deployment",
    
    # LLM settings (optional)
    llm_model="meta-llama/Llama-2-7b-chat-hf",  # HuggingFace repo
    # OR
    llm_model="/path/to/local/model.gguf",  # Local GGUF file
    
    # Performance
    use_gpu=True  # Enable GPU acceleration
)
```

## 📊 Understanding the Output

```python
result = agent.predict(conversation)

# result contains:
{
    'probability': 0.752,  # Conversion probability (0-1)
    'turn': 3,            # Current conversation turn
    'status': '🟢 High',  # Status indicator
    'metrics': {
        'customer_engagement': 0.8,
        'sales_effectiveness': 0.7,
        'conversation_length': 4.0,
        'outcome': 0.5,
        'progress': 0.15
    }
}
```

### Status Indicators
- 🟢 **High** (≥70%): Conversion highly likely
- 🟡 **Medium** (≥50%): Good potential
- 🟠 **Low** (≥30%): Needs improvement
- 🔴 **Very Low** (<30%): Unlikely to convert

## 🗂️ Conversation Format

The package accepts conversations in multiple formats:

### Simple List Format
```python
conversation = [
    "Customer message 1",
    "Sales rep response 1",
    "Customer message 2",
    "Sales rep response 2"
]
```

### Structured Format
```python
conversation = [
    {"speaker": "customer", "message": "I need help"},
    {"speaker": "sales_rep", "message": "I'm here to help!"}
]
```

### Alternative Keys
```python
# Also supports 'role' and 'content' keys
conversation = [
    {"role": "user", "content": "I need a CRM"},
    {"role": "assistant", "content": "Let me help you find the right solution"}
]
```

## 🛠️ Troubleshooting

### GPU Support

If you're having issues with GPU:

```python
import torch
print(torch.cuda.is_available())  # Should return True for GPU

# Force CPU usage if needed
agent = sales.Agent(use_gpu=False)
```

### Model Download Issues

```python
# Manually download model
from deepmost.core.utils import download_model

url = "https://github.com/DeepMostInnovations/sales-conversion-model-reinf-learning/raw/main/sales_conversion_model.zip"
path = "/path/to/save/model.zip"
download_model(url, path)

# Use the downloaded model
agent = sales.Agent(model_path=path, auto_download=False)
```

### LLM Memory Issues

For large language models on limited GPU memory:

```python
# Use smaller model or CPU offloading
agent = sales.Agent(
    llm_model="unsloth/tinyllama-1.1B-chat-GGUF",  # Smaller model
    use_gpu=True
)
```

## 📈 Performance Tips

1. **Batch Processing**: Process multiple conversations efficiently
```python
conversations = [conv1, conv2, conv3]
results = [agent.predict(conv) for conv in conversations]
```

2. **Persistent Agent**: Reuse the same agent instance
```python
agent = sales.Agent()  # Initialize once
for conv in conversations:
    result = agent.predict(conv)
```

3. **GPU Optimization**: Ensure CUDA is properly configured
```bash
# Check CUDA version
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

```bash
# Clone the repository
git clone https://github.com/DeepMostInnovations/deepmost.git
cd deepmost

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Embeddings by [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- LLM support via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/DeepMostInnovations/deepmost/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DeepMostInnovations/deepmost/discussions)
- **Email**: support@deepmost.ai

## 🔗 Links

- [PyPI Package](https://pypi.org/project/deepmost/)
- [GitHub Repository](https://github.com/DeepMostInnovations/deepmost)
- [Documentation](https://deepmost.readthedocs.io/)
- [Model Repository](https://huggingface.co/DeepMostInnovations)

---

Made with ❤️ by [DeepMost Innovations](https://deepmost.ai)