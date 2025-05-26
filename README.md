# DeepMost - Advanced Sales Conversation Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/deepmost.svg)](https://badge.fury.io/py/deepmost)

A powerful Python package for analyzing sales conversations and predicting conversion probability using advanced reinforcement learning. **DeepMost specializes in turn-by-turn conversation analysis**, showing you exactly how each message impacts your sales success.

## 🚀 Key Features

- **Turn-by-Turn Conversation Analysis**: Track how conversion probability evolves with each message exchange
- **Advanced PPO Reinforcement Learning**: Trained on real sales conversations for accurate predictions
- **Dual Backend Support**: Choose between open-source (HuggingFace + GGUF) or Azure OpenAI backends
- **Dynamic LLM-Powered Metrics**: Real-time analysis of customer engagement and sales effectiveness
- **Sales Training & Coaching**: Identify which conversation elements increase or decrease conversion probability
- **A/B Testing Sales Scripts**: Compare different approaches and optimize your sales methodology
- **Real-time Sales Assistance**: Get insights during live conversations to guide next steps
- **GPU Acceleration**: Full CUDA/Metal support for fast analysis (open-source backend)
- **Enterprise Ready**: Azure OpenAI integration for enterprise deployments

## 📦 Installation

### Requirements
- **Open-Source Backend**: Python 3.11+ (no other versions supported)
- **Azure Backend**: Python 3.10+ 

### Open-Source Installation (Recommended for Development)

**Basic Installation:**
```bash
pip install deepmost
```

**With GPU Support (Recommended):**
```bash
pip install deepmost[gpu]
```

**Manual GPU Setup (If automatic installation fails):**

*For NVIDIA CUDA:*
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
pip install deepmost
```

*For Apple Metal (M1/M2/M3):*
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
pip install deepmost
```

### Azure OpenAI Installation (Enterprise)

```bash
pip install deepmost
```

*Note: Azure backend doesn't require GPU compilation as it uses cloud-based embeddings and models.*

### Verify Installation

```python
import torch
from deepmost import sales

print(f"CUDA Available: {torch.cuda.is_available()}")
info = sales.get_system_info()
print(f"Supported Backends: {info['supported_backends']}")
```

## 🎯 Quick Start

### Simple Turn-by-Turn Analysis (Open-Source)

```python
from deepmost import sales

conversation = [
    "Hello, I'm looking for information on your new AI-powered CRM",
    "You've come to the right place! Our AI CRM helps increase sales efficiency. What challenges are you facing?",
    "We struggle with lead prioritization and follow-up timing",
    "Excellent! Our AI automatically analyzes leads and suggests optimal follow-up times. Would you like to see a demo?",
    "That sounds interesting. What's the pricing like?"
]

# Analyze conversation progression (prints results automatically)
results = sales.analyze_progression(conversation, llm_model="unsloth/Qwen3-4B-GGUF")
```

**Output:**
```
Turn 1 (customer): "Hello, I'm looking for information on your new AI-pow..." -> Probability: 0.1744
Turn 2 (sales_rep): "You've come to the right place! Our AI CRM helps increa..." -> Probability: 0.3292
Turn 3 (customer): "We struggle with lead prioritization and follow-up timing" -> Probability: 0.4156
Turn 4 (sales_rep): "Excellent! Our AI automatically analyzes leads and sugge..." -> Probability: 0.3908
Turn 5 (customer): "That sounds interesting. What's the pricing like?" -> Probability: 0.5234

Final Conversion Probability: 52.34%
Final Status: 🟢 High
```

### Azure OpenAI Backend Usage

```python
from deepmost import sales

# Initialize with Azure OpenAI credentials
agent = sales.Agent(
    azure_api_key="your-azure-api-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="your-embedding-deployment-name"
)

conversation = [
    {"speaker": "customer", "message": "I've been researching CRM solutions for our team"},
    {"speaker": "sales_rep", "message": "Great! What's driving your search for a new CRM?"},
    {"speaker": "customer", "message": "Our current system lacks automation and good reporting"},
    {"speaker": "sales_rep", "message": "Those are exactly the areas where our platform excels."}
]

# Get detailed turn-by-turn analysis
results = agent.analyze_conversation_progression(conversation, print_results=True)
```

## 🔧 Backend Configuration

### Open-Source Backend (HuggingFace + GGUF)

**Basic Configuration:**
```python
from deepmost import sales

agent = sales.Agent(
    # Embedding model from HuggingFace
    embedding_model="BAAI/bge-m3",  # Default: 1024-dim embeddings
    
    # GGUF LLM for comprehensive metrics (highly recommended)
    llm_model="unsloth/Qwen3-4B-GGUF",  # Recommended balance of quality vs performance
    
    # Performance options
    use_gpu=True,  # Enable GPU acceleration
    auto_download=True  # Auto-download models if not found
)
```

**Recommended GGUF Models:**
```python
# Balanced quality vs performance (recommended)
agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")
agent = sales.Agent(llm_model="unsloth/Llama-3.2-3B-Instruct-GGUF")

# Higher quality (requires more resources)
agent = sales.Agent(llm_model="unsloth/Llama-3.1-8B-Instruct-GGUF")

# Smaller models for limited resources
agent = sales.Agent(llm_model="microsoft/Phi-3-mini-4k-instruct-gguf")
```

**Custom PPO Models:**
```python
# Use your own trained PPO model
agent = sales.Agent(
    model_path="/path/to/your/ppo_model.zip",
    embedding_model="BAAI/bge-m3",  # Must match training setup
    llm_model="unsloth/Qwen3-4B-GGUF"
)
```

### Azure OpenAI Backend (Enterprise)

**Basic Azure Configuration:**
```python
from deepmost import sales

agent = sales.Agent(
    # Azure OpenAI credentials
    azure_api_key="your-azure-openai-api-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="your-embedding-deployment",  # e.g., "text-embedding-ada-002"
    
    # Optional: specify API version
    # azure_api_version="2023-12-01-preview"  # Default
)
```

**Azure Setup Requirements:**

1. **Azure OpenAI Resource**: Create an Azure OpenAI resource in your subscription
2. **Embedding Deployment**: Deploy an embedding model (recommended: `text-embedding-ada-002`)
3. **API Key & Endpoint**: Get your API key and endpoint from Azure portal

**Example Azure Deployment Setup:**
```bash
# Using Azure CLI to create embedding deployment
az cognitiveservices account deployment create \
  --resource-group "your-rg" \
  --name "your-openai-resource" \
  --deployment-name "text-embedding-ada-002" \
  --model-name "text-embedding-ada-002" \
  --model-version "2" \
  --model-format "OpenAI" \
  --scale-settings-scale-type "Standard"
```

**Advanced Azure Configuration:**
```python
agent = sales.Agent(
    azure_api_key="your-api-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="text-embedding-ada-002",
    azure_api_version="2023-12-01-preview",
    
    # Optional: Custom PPO model path
    model_path="/path/to/azure-compatible-model.zip",
    
    # GPU not needed for Azure backend (cloud-based)
    use_gpu=False
)
```

**Environment Variable Setup (Recommended):**
```python
import os

# Set environment variables
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "text-embedding-ada-002"

# Initialize with environment variables
agent = sales.Agent(
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)
```

### Backend Comparison

| Feature | Open-Source Backend | Azure Backend |
|---------|-------------------|---------------|
| **Cost** | Free (local compute) | Pay-per-API-call |
| **Setup** | More complex (GPU setup) | Simpler (cloud-based) |
| **Privacy** | Complete data privacy | Data sent to Azure |
| **Performance** | Depends on local hardware | Consistent cloud performance |
| **LLM Analysis** | Full GGUF model analysis | Basic heuristic analysis |
| **Scalability** | Limited by local resources | Highly scalable |
| **Offline** | Works offline | Requires internet |
| **Enterprise** | Good for development | Ideal for production |

## 📊 Understanding Results

### Turn-by-Turn Analysis Output

```python
{
    'turn': 1,                           # Turn number (1-indexed)
    'speaker': 'customer',               # Who spoke this turn
    'message': 'I need a CRM',          # The actual message
    'probability': 0.3456,              # Conversion probability after this turn
    'status': '🟠 Low',                 # Visual status indicator
    'metrics': {                        # Detailed analysis metrics
        'customer_engagement': 0.6,      # Customer engagement score (0-1)
        'sales_effectiveness': 0.4,      # Sales rep effectiveness score (0-1)
        'conversation_length': 3.0,      # Number of messages so far
        'progress': 0.15,                # Conversation progress indicator
        'conversation_style': 'direct_professional',
        'conversation_flow': 'standard_linear',
        'primary_customer_needs': ['efficiency', 'cost_reduction']
        # ... additional metrics
    }
}
```

### Status Indicators
- 🟢 **High** (≥50%): Strong conversion potential - focus on closing
- 🟡 **Medium** (≥40%): Good potential - build value and address concerns  
- 🟠 **Low** (≥30%): Needs improvement - re-engage or discover deeper needs
- 🔴 **Very Low** (<30%): Poor fit or major obstacles - consider re-qualifying

### Comprehensive Metrics (Open-Source Backend with LLM)

When using the open-source backend with a GGUF LLM model, you get enhanced metrics:

```python
{
    # Core PPO Model Metrics
    'customer_engagement': 0.7,         # LLM-analyzed engagement level
    'sales_effectiveness': 0.6,         # LLM-analyzed sales approach quality
    'conversation_length': 5.0,
    'progress': 0.25,
    
    # Enhanced Conversation Analysis
    'conversation_style': 'consultative_advisory',
    'conversation_flow': 'gradual_discovery', 
    'communication_channel': 'video_call',
    'primary_customer_needs': ['efficiency', 'integration', 'analytics'],
    
    # Advanced Behavioral Analytics
    'engagement_trend': 0.8,            # Increasing engagement
    'objection_count': 0.2,             # Low objection level
    'value_proposition_mentions': 0.7,   # Strong value communication
    'technical_depth': 0.6,             # Moderately technical discussion
    'urgency_level': 0.4,               # Some time considerations
    'competitive_context': 0.3,         # Limited competitive mentions
    'pricing_sensitivity': 0.5,         # Moderate price focus
    'decision_authority_signals': 0.8,   # High decision-making authority
    
    # Probability Evolution
    'probability_trajectory': {0: 0.15, 1: 0.28, 2: 0.35, 3: 0.42, 4: 0.51}
}
```

## 💡 Practical Use Cases

### 1. Sales Training & Coaching

Analyze real conversations to identify what works:

```python
from deepmost import sales

# Training conversation example
training_conversation = [
    {"speaker": "customer", "message": "I'm comparing different CRM vendors"},
    {"speaker": "sales_rep", "message": "Smart approach! What's most important to you in a CRM?"},
    {"speaker": "customer", "message": "Integration with our existing tools"},
    {"speaker": "sales_rep", "message": "We integrate with 200+ tools. Which specific ones do you use?"},
    {"speaker": "customer", "message": "Mainly Salesforce, HubSpot, and Slack"},
    {"speaker": "sales_rep", "message": "Perfect! We have native integrations for all three. Let me show you how seamless the data sync is."}
]

agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")
results = agent.analyze_conversation_progression(training_conversation)

# Identify which turns increased/decreased probability
for i, result in enumerate(results[1:], 1):
    prev_prob = results[i-1]['probability']
    curr_prob = result['probability']
    change = curr_prob - prev_prob
    trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
    print(f"Turn {i+1}: {trend} {change:+.3f} change")
```

### 2. A/B Testing Sales Scripts

Compare different response strategies:

```python
# Test different ways to handle pricing questions
script_a_conversation = [
    "I'm interested but need to know pricing first",
    "Our Pro plan is $99/month per user with all features included"
]

script_b_conversation = [
    "I'm interested but need to know pricing first", 
    "I'd love to get you accurate pricing! What's your team size and main requirements?"
]

# Test both scripts
agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")
results_a = agent.analyze_conversation_progression(script_a_conversation, print_results=False)
results_b = agent.analyze_conversation_progression(script_b_conversation, print_results=False)

print(f"Script A final probability: {results_a[-1]['probability']:.2%}")
print(f"Script B final probability: {results_b[-1]['probability']:.2%}")
print(f"Improvement: {(results_b[-1]['probability'] - results_a[-1]['probability']):.2%}")
```

### 3. Real-time Sales Assistance

Use during live conversations for guidance:

```python
# Analyze ongoing conversation
current_conversation = [
    {"speaker": "customer", "message": "Your solution looks expensive compared to competitors"},
    {"speaker": "sales_rep", "message": "I understand the investment concern. Let me break down the ROI..."}
]

results = agent.analyze_conversation_progression(current_conversation, print_results=False)

# Get trend and recommendations
if len(results) >= 2:
    trend_change = results[-1]['probability'] - results[-2]['probability']
    trend = "📈 Improving" if trend_change > 0 else "📉 Declining"
    print(f"Conversation trend: {trend} ({trend_change:+.3f})")

# Get AI-powered suggestions based on current state
current_metrics = results[-1]['metrics']
if current_metrics['customer_engagement'] < 0.5:
    print("💡 Suggestion: Customer engagement is low. Ask open-ended questions to re-engage.")
elif current_metrics['sales_effectiveness'] < 0.5:
    print("💡 Suggestion: Refine your approach. Focus on customer needs and value proposition.")
```

### 4. Enterprise Integration with Azure

For enterprise deployments with Azure OpenAI:

```python
import os
from deepmost import sales

# Enterprise configuration with environment variables
agent = sales.Agent(
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

def analyze_sales_call(conversation_data):
    """Analyze a sales call for enterprise reporting"""
    results = agent.analyze_conversation_progression(
        conversation_data, 
        print_results=False
    )
    
    return {
        'final_probability': results[-1]['probability'],
        'status': results[-1]['status'],
        'key_metrics': {
            'engagement': results[-1]['metrics']['customer_engagement'],
            'effectiveness': results[-1]['metrics']['sales_effectiveness'],
            'objections': results[-1]['metrics']['objection_count']
        },
        'recommended_actions': results[-1]['metrics'].get('suggested_action', 'Continue building rapport')
    }

# Use in production
call_analysis = analyze_sales_call(your_conversation_data)
```

### 5. Batch Processing for Analytics

Process multiple conversations for insights:

```python
conversations = [
    # Load your conversation datasets
    {"id": "conv_1", "messages": [...]},
    {"id": "conv_2", "messages": [...]},
    # ... more conversations
]

agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")
results = []

for conv in conversations:
    analysis = agent.analyze_conversation_progression(
        conv["messages"], 
        conversation_id=conv["id"],
        print_results=False
    )
    
    results.append({
        'conversation_id': conv["id"],
        'final_probability': analysis[-1]['probability'],
        'turn_count': len(analysis),
        'avg_engagement': np.mean([turn['metrics']['customer_engagement'] for turn in analysis]),
        'avg_effectiveness': np.mean([turn['metrics']['sales_effectiveness'] for turn in analysis])
    })

# Analyze results
import pandas as pd
df = pd.DataFrame(results)
print(f"Average conversion probability: {df['final_probability'].mean():.2%}")
print(f"High-performing conversations (>50%): {(df['final_probability'] > 0.5).sum()}")
```

## 📝 Conversation Formats

DeepMost accepts multiple conversation formats:

### Structured Format (Recommended)
```python
conversation = [
    {"speaker": "customer", "message": "I need help choosing a CRM"},
    {"speaker": "sales_rep", "message": "I'd be happy to help! What's your main challenge?"}
]
```

### Simple List Format
```python
conversation = [
    "I need help choosing a CRM",        # Assumed customer (odd positions)
    "I'd be happy to help! What's your main challenge?"  # Assumed sales_rep (even positions)
]
```

### OpenAI Chat Format
```python
conversation = [
    {"role": "user", "content": "I need a CRM"},
    {"role": "assistant", "content": "Let me help you find the right solution"}
]
```

**Supported speaker mappings:**
- Customer: `customer`, `user` 
- Sales Rep: `sales_rep`, `assistant`, `agent`, `bot`, `model`

## 🛠️ Troubleshooting

### Open-Source Backend Issues

**GPU Installation Problems:**
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test llama-cpp-python
try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python installed successfully")
except ImportError:
    print("❌ llama-cpp-python not installed")
```

**Manual GPU Setup:**
```bash
# Install CMake first
pip install cmake

# For NVIDIA CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# For Apple Metal
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Then install DeepMost
pip install deepmost
```

**LLM Model Issues:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed LLM outputs for troubleshooting
agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")
```

### Azure Backend Issues

**Authentication Problems:**
```python
# Test Azure connection
try:
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        api_key="your-api-key",
        azure_endpoint="https://your-resource.openai.azure.com",
        api_version="2023-12-01-preview"
    )
    
    # Test embedding call
    response = client.embeddings.create(
        input="test",
        model="your-deployment-name"  # Your embedding deployment
    )
    print("✅ Azure OpenAI connection successful")
    
except Exception as e:
    print(f"❌ Azure connection failed: {e}")
```

**Common Azure Issues:**
1. **Invalid API Key**: Check your Azure OpenAI resource API keys
2. **Wrong Endpoint**: Ensure endpoint format: `https://your-resource.openai.azure.com`
3. **Deployment Not Found**: Verify your embedding deployment name exists
4. **Quota Exceeded**: Check your Azure OpenAI usage quotas
5. **Region Issues**: Ensure your deployment region supports the embedding model

**Azure Configuration Validation:**
```python
def validate_azure_config():
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        return False
    
    print("✅ All required Azure environment variables set")
    return True

validate_azure_config()
```

## 📈 Performance Optimization

### Open-Source Backend

**Best Practices:**
1. **Reuse Agent**: Initialize once, use multiple times
2. **GPU Memory**: Monitor with `nvidia-smi` (CUDA) or Activity Monitor (Metal)
3. **Model Size**: Balance quality vs. performance needs
4. **Batch Processing**: Process multiple conversations efficiently

**Memory Management:**
```python
# For limited GPU memory, use smaller models
agent = sales.Agent(
    llm_model="microsoft/Phi-3-mini-4k-instruct-gguf",  # Smaller model
    use_gpu=True
)

# Monitor GPU memory usage
import torch
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
```

### Azure Backend

**Cost Optimization:**
```python
# Batch multiple predictions to reduce API calls
conversations_batch = [conv1, conv2, conv3, ...]

# Process efficiently
agent = sales.Agent(
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

results = []
for conv in conversations_batch:
    result = agent.analyze_conversation_progression(conv, print_results=False)
    results.append(result)
```

**Rate Limiting:**
```python
import time
from typing import List

def batch_analyze_with_rate_limit(agent, conversations: List, delay: float = 1.0):
    """Analyze conversations with rate limiting for Azure API"""
    results = []
    
    for i, conv in enumerate(conversations):
        try:
            result = agent.analyze_conversation_progression(conv, print_results=False)
            results.append(result)
            
            # Rate limiting
            if i < len(conversations) - 1:  # Don't delay after last item
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error processing conversation {i}: {e}")
            continue
    
    return results
```

## 🔄 Migration Between Backends

### From Open-Source to Azure

```python
# Original open-source setup
agent_os = sales.Agent(
    llm_model="unsloth/Qwen3-4B-GGUF",
    use_gpu=True
)

# Migrate to Azure
agent_azure = sales.Agent(
    azure_api_key="your-api-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="text-embedding-ada-002"
)

# Same conversation analysis API
conversation = [...]  # Your conversation data
results_os = agent_os.analyze_conversation_progression(conversation)
results_azure = agent_azure.analyze_conversation_progression(conversation)
```

### Hybrid Approach

```python
def get_agent(use_azure: bool = False):
    """Factory function for backend selection"""
    if use_azure:
        return sales.Agent(
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )
    else:
        return sales.Agent(
            llm_model="unsloth/Qwen3-4B-GGUF",
            use_gpu=True
        )

# Use based on environment or requirements
agent = get_agent(use_azure=os.getenv("USE_AZURE_BACKEND", "false").lower() == "true")
```

## 🤝 Contributing

We welcome contributions! Focus areas:
- Enhanced conversation analysis metrics
- Additional LLM model support  
- Integration with popular sales tools
- Performance optimizations
- Azure OpenAI enhancements

```bash
git clone https://github.com/DeepMostInnovations/deepmost.git
cd deepmost
pip install -e .[dev]
pytest tests/
```

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/DeepMostInnovations/deepmost.git
cd deepmost

# Install with development dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run code formatting
black deepmost/
isort deepmost/
flake8 deepmost/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PPO Training**: [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) & [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- **LLM Support**: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- **Models**: [HuggingFace](https://huggingface.co/) & [Unsloth](https://github.com/unslothai/unsloth)

## 📞 Support & Links

- **Documentation**: [https://deepmost.readthedocs.io/](https://deepmost.readthedocs.io/)
- **GitHub Issues**: [https://github.com/DeepMostInnovations/deepmost/issues](https://github.com/DeepMostInnovations/deepmost/issues)
- **PyPI Package**: [https://pypi.org/project/deepmost/](https://pypi.org/project/deepmost/)
- **Model Repository**: [https://huggingface.co/DeepMostInnovations](https://huggingface.co/DeepMostInnovations)
- **Email Support**: support@deepmostai.com

## 🚀 Getting Started Checklist

### For Development/Testing (Open-Source)
- [ ] Install Python 3.11+
- [ ] Run `pip install deepmost[gpu]`
- [ ] Verify GPU setup with `torch.cuda.is_available()`
- [ ] Test with simple conversation using `sales.analyze_progression()`

### For Production/Enterprise (Azure)
- [ ] Create Azure OpenAI resource
- [ ] Deploy embedding model (text-embedding-ada-002 recommended)
- [ ] Set environment variables for API credentials
- [ ] Install DeepMost: `pip install deepmost`
- [ ] Test Azure connection and run analysis

---

**Transform your sales conversations into actionable insights. Start analyzing what drives conversions today!** 🎯

Made with ❤️ by [DeepMost Innovations](https://www.deepmostai.com/)