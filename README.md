# DeepMost - Advanced Sales Conversation Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python package for analyzing sales conversations and predicting conversion probability using advanced reinforcement learning. **DeepMost specializes in turn-by-turn conversation analysis**, showing you exactly how each message impacts your sales success.

## 🚀 Key Features

- **Turn-by-Turn Conversation Analysis**: Track how conversion probability evolves with each message exchange
- **Advanced PPO Reinforcement Learning**: Trained on real sales conversations for accurate predictions
- **Triple Backend Support**: Choose between Open-source (HuggingFace + GGUF), Azure OpenAI, or Standard OpenAI
- **Dynamic LLM-Powered Metrics**: Real-time analysis of customer engagement and sales effectiveness
- **Sales Training & Coaching**: Identify which conversation elements increase or decrease conversion probability
- **A/B Testing Sales Scripts**: Compare different approaches and optimize your sales methodology
- **Real-time Sales Assistance**: Get insights during live conversations to guide next steps
- **GPU Acceleration**: Full CUDA/Metal support for fast analysis (open-source backend)
- **Enterprise Ready**: Azure OpenAI and Standard OpenAI integration for production deployments

## 📦 Installation

### Requirements
- **Open-Source Backend**: Python 3.11+ (no other versions supported)
- **Azure/Standard OpenAI Backends**: Python 3.10+ 

### Basic Installation
```bash
pip install deepmost
```

### Open-Source Backend with GPU Support
For best performance and local LLM analysis:
```bash
pip install deepmost[gpu]
```

### Manual GPU Setup (If automatic installation fails)

**For NVIDIA CUDA:**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
pip install deepmost
```

**For Apple Metal (M1/M2/M3):**
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
pip install deepmost
```

### Verify Installation
```python
import torch
from deepmost import sales

print(f"CUDA Available: {torch.cuda.is_available()}")
info = sales.get_system_info()
print(f"Supported Backends: {info['supported_backends']}")
```

## 🎯 Quick Start Examples

### Open-Source Backend (Local & Private)
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
Backend: Opensource
```

### Standard OpenAI Backend (Latest Models)
```python
from deepmost import sales

# Initialize with standard OpenAI (latest models)
agent = sales.Agent(
    openai_api_key="your-openai-api-key",
    openai_embedding_model="text-embedding-3-large",  # Latest embedding model
    openai_chat_model="gpt-4o"  # Latest chat model
)

conversation = [
    {"speaker": "customer", "message": "I've been researching CRM solutions for our team"},
    {"speaker": "sales_rep", "message": "Great! What's driving your search for a new CRM?"},
    {"speaker": "customer", "message": "Our current system lacks automation and good reporting"},
    {"speaker": "sales_rep", "message": "Those are exactly the areas where our platform excels."}
]

# Get detailed turn-by-turn analysis with full LLM-powered metrics
results = agent.analyze_conversation_progression(conversation, print_results=True)
```

### Azure OpenAI Backend (Enterprise)
```python
from deepmost import sales

# Initialize with Azure OpenAI (enterprise security)
agent = sales.Agent(
    azure_api_key="your-azure-api-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="text-embedding-3-large",  # Embedding deployment
    azure_chat_deployment="gpt-4o"  # Chat completion deployment
)

conversation = [
    {"speaker": "customer", "message": "We need a solution that integrates with our existing tools"},
    {"speaker": "sales_rep", "message": "Our platform offers native integrations with 200+ tools. Which ones are most important to you?"}
]

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

### Standard OpenAI Backend (Latest Models)

**Basic Configuration:**
```python
from deepmost import sales

agent = sales.Agent(
    # Standard OpenAI API key
    openai_api_key="your-openai-api-key",
    
    # Latest embedding models (choose based on needs)
    openai_embedding_model="text-embedding-3-large",  # High performance (3072 dims)

    
    # Latest chat models
    openai_chat_model="gpt-4o",  # Best performance
    # openai_chat_model="gpt-4o-mini",  # Cost-effective
)
```

**OpenAI Model Options:**
```python
# High performance setup (recommended for production)
agent = sales.Agent(
    openai_api_key="your-api-key",
    openai_embedding_model="text-embedding-3-large",  # 3072 dimensions
    openai_chat_model="gpt-4o"  # Latest GPT-4o
)

# Cost-effective setup
agent = sales.Agent(
    openai_api_key="your-api-key",
    openai_embedding_model="text-embedding-3-large", 
    openai_chat_model="gpt-4o-mini"  # Smaller, faster, cheaper
)

# Legacy models (still supported)
agent = sales.Agent(
    openai_api_key="your-api-key",
    openai_embedding_model="text-embedding-3-large",  # Original model
    openai_chat_model="gpt-3.5-turbo"  # GPT-3.5
)
```

### Azure OpenAI Backend (Enterprise)

**Basic Configuration:**
```python
from deepmost import sales

agent = sales.Agent(
    # Azure OpenAI credentials
    azure_api_key="your-azure-openai-api-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="text-embedding-3-large",  # Embedding deployment
    azure_chat_deployment="gpt-4o",  # Chat completion deployment
    
    # Optional: specify API version (default: "2024-10-21")
    azure_api_version="2024-10-21"
)
```

**Azure Setup Requirements:**

1. **Azure OpenAI Resource**: Create an Azure OpenAI resource in your subscription
2. **Embedding Deployment**: Deploy an embedding model (recommended: `text-embedding-3-large`)
3. **Chat Deployment**: Deploy a chat model (recommended: `gpt-4o`, `gpt-4o-mini`, or `gpt-35-turbo`)
4. **API Key & Endpoint**: Get your API key and endpoint from Azure portal

**Example Azure Deployment Setup:**
```bash
# Using Azure CLI to create deployments
# 1. Create embedding deployment
az cognitiveservices account deployment create \
  --resource-group "your-rg" \
  --name "your-openai-resource" \
  --deployment-name "text-embedding-3-large" \
  --model-name "text-embedding-3-large" \
  --model-version "2" \
  --model-format "OpenAI" \
  --scale-settings-scale-type "Standard"

# 2. Create chat completion deployment
az cognitiveservices account deployment create \
  --resource-group "your-rg" \
  --name "your-openai-resource" \
  --deployment-name "gpt-4o" \
  --model-name "gpt-4o" \
  --model-version "2024-08-06" \
  --model-format "OpenAI" \
  --scale-settings-scale-type "Standard"
```

**Azure Chat Model Options:**
```python
# High performance (recommended for production)
agent = sales.Agent(
    azure_chat_deployment="gpt-4o",  # Latest GPT-4o
    # ... other Azure config
)

# Cost-effective option
agent = sales.Agent(
    azure_chat_deployment="gpt-4o-mini",  # Smaller, faster, cheaper
    # ... other Azure config
)

# Legacy option (still supported)
agent = sales.Agent(
    azure_chat_deployment="gpt-35-turbo",  # GPT-3.5 Turbo
    # ... other Azure config
)
```

### Backend Comparison

| Feature | Open-Source | Standard OpenAI | Azure OpenAI |
|---------|-------------|-----------------|--------------|
| **Cost** | Free (local compute) | Pay-per-API-call | Pay-per-API-call |
| **Setup** | More complex (GPU) | Simple (API key) | Moderate (deployments) |
| **Privacy** | Complete data privacy | Data sent to OpenAI | Data sent to Azure |
| **Performance** | Depends on hardware | Consistent cloud | Consistent cloud |
| **Latest Models** | Limited to GGUF | ✅ Latest GPT-4o, embeddings | ✅ Enterprise versions |
| **LLM Analysis** | ✅ Full local analysis | ✅ Full cloud analysis | ✅ Full cloud analysis |
| **Response Generation** | ✅ Full capabilities | ✅ Full capabilities | ✅ Full capabilities |
| **Scalability** | Limited by hardware | Highly scalable | Highly scalable |
| **Offline** | ✅ Works offline | ❌ Requires internet | ❌ Requires internet |
| **Enterprise** | Good for development | Good for startups | ✅ Ideal for enterprise |
| **Compliance** | Self-managed | OpenAI terms | ✅ Enterprise compliance |

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
        # ... additional comprehensive metrics
    }
}
```

### Status Indicators
- 🟢 **High** (≥50%): Strong conversion potential - focus on closing
- 🟡 **Medium** (≥40%): Good potential - build value and address concerns  
- 🟠 **Low** (≥30%): Needs improvement - re-engage or discover deeper needs
- 🔴 **Very Low** (<30%): Poor fit or major obstacles - consider re-qualifying

### Comprehensive Metrics (All Backends with LLM)

When using any backend with LLM support enabled, you get enhanced metrics:

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

# Use any backend (example with OpenAI)
agent = sales.Agent(
    openai_api_key="your-api-key",
    openai_chat_model="gpt-4o"
)

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

Compare different response strategies across backends:

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

# Test with different backends
backends = [
    {"openai_api_key": "key", "openai_chat_model": "gpt-4o"},
    {"azure_api_key": "key", "azure_endpoint": "endpoint", "azure_deployment": "embedding", "azure_chat_deployment": "gpt-4o"},
    {"llm_model": "unsloth/Qwen3-4B-GGUF"}
]

for i, backend_config in enumerate(backends):
    agent = sales.Agent(**backend_config)
    results_a = agent.analyze_conversation_progression(script_a_conversation, print_results=False)
    results_b = agent.analyze_conversation_progression(script_b_conversation, print_results=False)
    
    backend_name = ["OpenAI", "Azure", "Open-source"][i]
    print(f"\n{backend_name} Backend:")
    print(f"Script A final probability: {results_a[-1]['probability']:.2%}")
    print(f"Script B final probability: {results_b[-1]['probability']:.2%}")
    print(f"Improvement: {(results_b[-1]['probability'] - results_a[-1]['probability']):.2%}")
```

### 3. Real-time Sales Assistance

Use during live conversations for guidance:

```python
# Analyze ongoing conversation with response generation
current_conversation = [
    {"speaker": "customer", "message": "Your solution looks expensive compared to competitors"},
    {"speaker": "sales_rep", "message": "I understand the investment concern. Let me break down the ROI..."}
]

user_message = "I'm still not convinced it's worth the price difference"

# Generate intelligent response and get predictions
agent = sales.Agent(openai_api_key="your-key", openai_chat_model="gpt-4o")

response_result = agent.predict_with_response(
    conversation=current_conversation,
    user_input=user_message,
    system_prompt="You are a professional sales representative focused on value-based selling."
)

print(f"Suggested Response: {response_result['response']}")
print(f"Predicted Probability: {response_result['prediction']['probability']:.2%}")
print(f"Status: {response_result['prediction']['status']}")

# Get comprehensive metrics
metrics = response_result['prediction']['metrics']
if metrics['customer_engagement'] < 0.5:
    print("💡 Suggestion: Customer engagement is low. Ask open-ended questions to re-engage.")
elif metrics['pricing_sensitivity'] > 0.7:
    print("💡 Suggestion: High price sensitivity detected. Focus on ROI and value demonstration.")
```

### 4. Enterprise Integration Examples

#### Standard OpenAI Integration
```python
import os
from deepmost import sales

# Production configuration with environment variables
agent = sales.Agent(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_embedding_model="text-embedding-3-large",
    openai_chat_model="gpt-4o"
)

def analyze_sales_call_with_response_generation(conversation_data, user_input):
    """Enterprise sales call analysis with AI-generated responses"""
    
    # Generate intelligent response and predict outcome
    response_result = agent.predict_with_response(
        conversation=conversation_data,
        user_input=user_input,
        system_prompt="You are a professional sales representative focused on understanding customer needs and building value."
    )
    
    # Analyze full conversation progression
    full_conversation = conversation_data + [
        {"speaker": "customer", "message": user_input},
        {"speaker": "sales_rep", "message": response_result['response']}
    ]
    
    progression_results = agent.analyze_conversation_progression(
        full_conversation, 
        print_results=False
    )
    
    return {
        'generated_response': response_result['response'],
        'final_probability': progression_results[-1]['probability'],
        'status': progression_results[-1]['status'],
        'comprehensive_metrics': {
            'engagement': progression_results[-1]['metrics']['customer_engagement'],
            'effectiveness': progression_results[-1]['metrics']['sales_effectiveness'],
            'conversation_style': progression_results[-1]['metrics']['conversation_style'],
            'objection_level': progression_results[-1]['metrics']['objection_count'],
            'technical_depth': progression_results[-1]['metrics']['technical_depth'],
            'urgency_signals': progression_results[-1]['metrics']['urgency_level'],
            'decision_authority': progression_results[-1]['metrics']['decision_authority_signals']
        },
        'recommended_actions': progression_results[-1]['metrics'].get('suggested_action', 'Continue building rapport'),
        'probability_evolution': [turn['probability'] for turn in progression_results],
        'backend_used': 'openai'
    }
```

#### Azure OpenAI Enterprise Integration
```python
import os
from deepmost import sales

# Enterprise Azure configuration
agent = sales.Agent(
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
)

def enterprise_conversation_analysis(conversation_data):
    """Enterprise-grade conversation analysis with Azure OpenAI"""
    results = agent.analyze_conversation_progression(conversation_data, print_results=False)
    
    return {
        'conversion_probability': results[-1]['probability'],
        'confidence_level': results[-1]['status'],
        'key_insights': {
            'customer_sentiment': results[-1]['metrics']['customer_engagement'],
            'sales_approach_quality': results[-1]['metrics']['sales_effectiveness'],
            'conversation_complexity': results[-1]['metrics']['technical_depth'],
            'purchase_readiness': results[-1]['metrics']['decision_authority_signals']
        },
        'recommended_next_steps': results[-1]['metrics'].get('suggested_action'),
        'backend_compliance': 'azure_enterprise'
    }
```

### 5. Multi-Backend Comparison Analysis

```python
def compare_backends_analysis(conversation):
    """Compare analysis across all three backends"""
    
    backends = {
        'Open-source': sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF"),
        'OpenAI': sales.Agent(openai_api_key="key", openai_chat_model="gpt-4o"),
        'Azure': sales.Agent(azure_api_key="key", azure_endpoint="endpoint", 
                           azure_deployment="embedding", azure_chat_deployment="gpt-4o")
    }
    
    results = {}
    
    for backend_name, agent in backends.items():
        try:
            analysis = agent.analyze_conversation_progression(conversation, print_results=False)
            results[backend_name] = {
                'final_probability': analysis[-1]['probability'],
                'engagement_score': analysis[-1]['metrics']['customer_engagement'],
                'effectiveness_score': analysis[-1]['metrics']['sales_effectiveness'],
                'conversation_style': analysis[-1]['metrics']['conversation_style']
            }
        except Exception as e:
            results[backend_name] = {'error': str(e)}
    
    return results

# Usage
conversation = [
    {"speaker": "customer", "message": "We're evaluating CRM solutions"},
    {"speaker": "sales_rep", "message": "What's driving your evaluation?"},
    {"speaker": "customer", "message": "Need better reporting and automation"}
]

comparison = compare_backends_analysis(conversation)
for backend, result in comparison.items():
    if 'error' not in result:
        print(f"{backend}: {result['final_probability']:.2%} probability, "
              f"Engagement: {result['engagement_score']:.2f}")
```

## 📝 Conversation Formats

DeepMost accepts multiple conversation formats across all backends:

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

### Standard OpenAI Backend Issues

**Authentication Problems:**
```python
# Test OpenAI connection
try:
    from openai import OpenAI
    
    client = OpenAI(api_key="your-api-key")
    
    # Test embedding call
    response = client.embeddings.create(
        input="test",
        model="text-embedding-3-large"
    )
    print("✅ OpenAI connection successful")
    
    # Test chat call if needed
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("✅ OpenAI chat completion successful")
    
except Exception as e:
    print(f"❌ OpenAI connection failed: {e}")
```

**Common OpenAI Issues:**
1. **Invalid API Key**: Check your OpenAI API key
2. **Model Not Available**: Ensure you have access to the requested models
3. **Rate Limits**: Check your usage quotas and rate limits
4. **Billing Issues**: Ensure your OpenAI account has sufficient credits

### Azure OpenAI Backend Issues

**Authentication Problems:**
```python
# Test Azure connection
try:
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        api_key="your-api-key",
        azure_endpoint="https://your-resource.openai.azure.com",
        api_version="2024-10-21"
    )
    
    # Test embedding call
    response = client.embeddings.create(
        input="test",
        model="your-deployment-name"
    )
    print("✅ Azure OpenAI connection successful")
    
except Exception as e:
    print(f"❌ Azure connection failed: {e}")
```

**Common Azure Issues:**
1. **Invalid API Key**: Check your Azure OpenAI resource API keys
2. **Wrong Endpoint**: Ensure endpoint format: `https://your-resource.openai.azure.com`
3. **Deployment Not Found**: Verify your embedding deployment name exists
4. **Chat Deployment Issues**: Ensure your chat deployment exists and model is deployed
5. **Quota Exceeded**: Check your Azure OpenAI usage quotas
6. **Region Issues**: Ensure your deployment region supports the embedding and chat models
7. **API Version Mismatch**: Use supported API version (default: "2024-10-21")

**Configuration Validation:**
```python
def validate_all_backends():
    """Validate all backend configurations"""
    
    # Test Open-source
    try:
        agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")
        print("✅ Open-source backend available")
    except Exception as e:
        print(f"❌ Open-source backend error: {e}")
    
    # Test OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            agent = sales.Agent(
                openai_api_key=openai_key,
                openai_chat_model="gpt-4o"
            )
            print("✅ OpenAI backend available")
        except Exception as e:
            print(f"❌ OpenAI backend error: {e}")
    else:
        print("⚠️ OPENAI_API_KEY not set")
    
    # Test Azure
    azure_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    if all(os.getenv(var) for var in azure_vars):
        try:
            agent = sales.Agent(
                azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            )
            print("✅ Azure OpenAI backend available")
        except Exception as e:
            print(f"❌ Azure backend error: {e}")
    else:
        print("⚠️ Azure environment variables not set")

validate_all_backends()
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

### Standard OpenAI Backend


```python
# Use cost-effective models for batch processing
agent = sales.Agent(
    openai_api_key="your-key",
    openai_embedding_model="text-embedding-3-large",  # Cheaper than 3-large
    openai_chat_model="gpt-4o-mini"  # Cheaper than gpt-4o
)

# Batch processing for efficiency
conversations = [conv1, conv2, conv3, ...]
results = []

for conv in conversations:
    result = agent.analyze_conversation_progression(conv, print_results=False)
    results.append(result)
    # Optional: add small delay to avoid rate limits
    time.sleep(0.1)
```

### Azure OpenAI Backend

**Enterprise Optimization:**
```python
# Configure for enterprise scale
agent = sales.Agent(
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
)

# Rate limiting for Azure API
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
            if i < len(conversations) - 1:
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error processing conversation {i}: {e}")
            continue
    
    return results
```

## 🔄 Migration Between Backends

### Backend Flexibility

```python
def get_agent(backend_preference: str = "auto"):
    """Factory function for flexible backend selection"""
    
    if backend_preference == "openai" and os.getenv("OPENAI_API_KEY"):
        return sales.Agent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_embedding_model="text-embedding-3-large",
            openai_chat_model="gpt-4o"
        )
    elif backend_preference == "azure" and all([
        os.getenv("AZURE_OPENAI_API_KEY"),
        os.getenv("AZURE_OPENAI_ENDPOINT"),
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    ]):
        return sales.Agent(
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        )
    else:
        # Fallback to open-source
        return sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")

# Use based on environment or requirements
agent = get_agent(backend_preference="openai")
```

### Environment-Based Configuration

```python
# Environment variables for all backends
# .env file example:

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_CHAT_MODEL=gpt-4o

# Azure Configuration  
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o

# Open-source Configuration
OPENSOURCE_LLM_MODEL=unsloth/Qwen3-4B-GGUF
USE_GPU=true

# Backend Selection
DEEPMOST_BACKEND=openai  # or azure, opensource
```

```python
import os
from deepmost import sales

def create_agent_from_env():
    """Create agent based on environment configuration"""
    backend = os.getenv("DEEPMOST_BACKEND", "auto").lower()
    
    if backend == "openai":
        return sales.Agent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        )
    elif backend == "azure":
        return sales.Agent(
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        )
    elif backend == "opensource":
        return sales.Agent(
            llm_model=os.getenv("OPENSOURCE_LLM_MODEL", "unsloth/Qwen3-4B-GGUF"),
            use_gpu=os.getenv("USE_GPU", "true").lower() == "true"
        )
    else:
        # Auto-select based on available credentials
        if os.getenv("OPENAI_API_KEY"):
            return create_agent_from_env.__wrapped__()  # Retry with openai
        elif all([os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_ENDPOINT")]):
            return create_agent_from_env.__wrapped__()  # Retry with azure
        else:
            return sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")

agent = create_agent_from_env()
```

## 🤝 Contributing

We welcome contributions! Focus areas:
- Enhanced conversation analysis metrics
- Additional LLM model support  
- Integration with popular sales tools
- Performance optimizations
- New backend implementations

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

# Run tests for all backends
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
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/), [OpenAI](https://openai.com/), & [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- **LLM Support**: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) & [OpenAI API](https://platform.openai.com/)
- **Models**: [HuggingFace](https://huggingface.co/) & [Unsloth](https://github.com/unslothai/unsloth)

## 📞 Support & Links

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

### For Production (OpenAI)
- [ ] Get OpenAI API key from [platform.openai.com](https://platform.openai.com)
- [ ] Set `OPENAI_API_KEY` environment variable
- [ ] Install DeepMost: `pip install deepmost`
- [ ] Test connection and run analysis

### For Enterprise (Azure OpenAI)
- [ ] Create Azure OpenAI resource
- [ ] Deploy embedding model (`text-embedding-3-large`)
- [ ] Deploy chat model (`gpt-4o` or `gpt-35-turbo`)
- [ ] Set Azure environment variables
- [ ] Install DeepMost: `pip install deepmost`
- [ ] Test Azure connection and run analysis

---

**Transform your sales conversations into actionable insights with three powerful backend options. Choose the approach that fits your needs!** 🎯

Made with ❤️ by [DeepMost Innovations](https://www.deepmostai.com/)