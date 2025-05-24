# DeepMost - Sales Conversation Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python package for analyzing sales conversations and tracking conversion probability evolution using advanced reinforcement learning. **DeepMost specializes in turn-by-turn conversation analysis**, showing you exactly how each message impacts your sales success.

## 🚀 Key Features

- **Turn-by-Turn Conversation Analysis**: Track how conversion probability evolves with each message exchange
- **Advanced PPO Reinforcement Learning**: Trained on real sales conversations for accurate predictions
- **Dynamic LLM-Powered Metrics**: Uses GGUF models to analyze customer engagement and sales effectiveness in real-time
- **Sales Training & Coaching**: Identify which conversation elements increase or decrease conversion probability
- **A/B Testing Sales Scripts**: Compare different approaches and optimize your sales methodology
- **Real-time Sales Assistance**: Get insights during live conversations to guide next steps
- **GPU Acceleration**: Full CUDA/Metal support for fast analysis
- **Open-Source Backend**: Leverages HuggingFace models and local GGUF LLMs

## 📦 Installation

Requires **Python 3.11** (does not support other versions).

### Recommended Installation (GPU Support)
For best performance and LLM-powered metrics:

```bash
pip install deepmost[gpu]
```

### Manual GPU Setup (If automatic fails)

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

## 🎯 Quick Start - Turn-by-Turn Analysis

### Simple Analysis (Recommended Usage)
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
Final Status: 🟡 Medium
```

### Detailed Analysis with Agent
```python
from deepmost import sales

# Initialize agent with LLM for best accuracy
agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")

conversation = [
    {"speaker": "customer", "message": "I've been researching CRM solutions for our team"},
    {"speaker": "sales_rep", "message": "Great! What's driving your search for a new CRM?"},
    {"speaker": "customer", "message": "Our current system lacks automation and good reporting"},
    {"speaker": "sales_rep", "message": "Those are exactly the areas where our platform excels. Our automation saves teams 15+ hours per week."}
]

# Get detailed turn-by-turn analysis
progression_results = agent.analyze_conversation_progression(conversation, print_results=True)

# Access detailed metrics for each turn
for turn_data in progression_results:
    print(f"\nTurn {turn_data['turn']}: {turn_data['probability']:.3f}")
    print(f"  Customer Engagement: {turn_data['metrics']['customer_engagement']:.2f}")
    print(f"  Sales Effectiveness: {turn_data['metrics']['sales_effectiveness']:.2f}")
```

## 📊 Understanding Turn-by-Turn Results

Each turn analysis provides:

```python
{
    'turn': 1,                           # Turn number (1-indexed)
    'speaker': 'customer',               # Who spoke this turn
    'message': 'I need a CRM',          # The actual message
    'probability': 0.3456,              # Conversion probability after this turn
    'status': '🟠 Low',                 # Visual status indicator
    'metrics': {                        # Detailed analysis metrics
        'customer_engagement': 0.6,      # LLM-derived engagement score
        'sales_effectiveness': 0.4,      # LLM-derived effectiveness score
        'conversation_length': 3.0,      # Number of messages so far
        'progress': 0.15                 # Conversation progress indicator
    }
}
```

### Status Indicators
- 🟢 **High** (≥70%): Strong conversion potential - focus on closing
- 🟡 **Medium** (≥50%): Good potential - build value and address concerns  
- 🟠 **Low** (≥30%): Needs improvement - re-engage or discover deeper needs
- 🔴 **Very Low** (<30%): Poor fit or major obstacles - consider re-qualifying

## 💡 Practical Use Cases

### 1. Sales Training & Coaching
Analyze real conversations to identify what works and what doesn't:

```python
training_conversation = [
    {"speaker": "customer", "message": "I'm comparing different CRM vendors"},
    {"speaker": "sales_rep", "message": "Smart approach! What's most important to you in a CRM?"},
    {"speaker": "customer", "message": "Integration with our existing tools"},
    {"speaker": "sales_rep", "message": "We integrate with 200+ tools. Which specific ones do you use?"},
    {"speaker": "customer", "message": "Mainly Salesforce, HubSpot, and Slack"},
    {"speaker": "sales_rep", "message": "Perfect! We have native integrations for all three. Let me show you how seamless the data sync is."}
]

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

results_a = sales.analyze_progression(script_a_conversation, llm_model="unsloth/Qwen3-4B-GGUF")
results_b = sales.analyze_progression(script_b_conversation, llm_model="unsloth/Qwen3-4B-GGUF")

print(f"Script A final probability: {results_a[-1]['probability']:.2%}")
print(f"Script B final probability: {results_b[-1]['probability']:.2%}")
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

### 4. Conversation Trend Visualization
```python
import matplotlib.pyplot as plt

# Analyze multiple scenarios
scenarios = {
    "Successful Sale": [
        "I need a CRM for my team",
        "What size team and main challenges?", 
        "10 people, need better lead tracking",
        "Our Pro plan is perfect for that. Here's how it works...",
        "Looks great! What's the next step?"
    ],
    "Price Objection": [
        "I need a CRM for my team",
        "Our premium solution is $99 per user monthly",
        "That's way too expensive",
        "Let me show you the ROI calculation...",
        "Still too much for our budget"
    ]
}

agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")

plt.figure(figsize=(12, 6))
for scenario_name, conversation in scenarios.items():
    results = agent.analyze_conversation_progression(conversation, print_results=False)
    probabilities = [r['probability'] for r in results]
    turns = list(range(1, len(probabilities) + 1))
    
    plt.plot(turns, probabilities, marker='o', linewidth=2, label=scenario_name)

plt.xlabel('Conversation Turn')
plt.ylabel('Conversion Probability')
plt.title('Conversion Probability Evolution by Scenario')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.show()
```

## 🔧 Configuration & Models

### Recommended LLM Models
For best accuracy with dynamic metrics:

```python
# Recommended models (balance of quality vs performance)
agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")          # Recommended
agent = sales.Agent(llm_model="unsloth/Llama-3.2-3B-Instruct-GGUF")  # Alternative

# For higher quality (requires more resources)
agent = sales.Agent(llm_model="unsloth/Llama-3.1-8B-Instruct-GGUF")


```

### Advanced Configuration
```python
agent = sales.Agent(
    # PPO Model (auto-downloads if not specified)
    model_path="/path/to/custom/ppo_model.zip",  # Optional custom model
    auto_download=True,
    
    # Embeddings (for conversation understanding)
    embedding_model="BAAI/bge-m3",  # Default: 1024-dim embeddings
    
    # LLM for dynamic metrics (highly recommended)
    llm_model="unsloth/Qwen3-4B-GGUF",
    
    # Performance
    use_gpu=True  # Enable GPU acceleration
)
```

### Using Custom PPO Models
```python
# If you've trained your own PPO model
agent = sales.Agent(
    model_path="/path/to/your/ppo_model.zip",
    embedding_model="BAAI/bge-m3",  # Must match your training setup
    llm_model="unsloth/Qwen3-4B-GGUF"
)
```

## 📝 Conversation Format

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

### GPU Installation Issues

**Check GPU Support:**
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

### LLM Model Issues

**Model Download Problems:**
- Models are cached in `~/.cache/huggingface/hub`
- Ensure stable internet connection and sufficient disk space
- Try smaller models if memory limited


**Debugging LLM Issues:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed LLM outputs for troubleshooting
agent = sales.Agent(llm_model="unsloth/Qwen3-4B-GGUF")
```

### Performance Optimization

**Best Practices:**
1. **Reuse Agent**: Initialize once, use multiple times
2. **GPU Memory**: Monitor with `nvidia-smi` (CUDA) or Activity Monitor (Metal)
3. **Batch Analysis**: Process multiple conversations efficiently
4. **Model Size**: Balance quality vs. performance needs

**Memory Management:**
```python
# For limited GPU memory
agent = sales.Agent(
    llm_model="unsloth/Qwen3-4B-GGUF",
    use_gpu=True
)


```

## 📈 Performance Tips

1. **Initialize Once**: Create agent instance once and reuse for multiple analyses
2. **Appropriate Model Size**: Use 3-4B parameter models for good balance of quality/speed
3. **GPU Acceleration**: Enable GPU for 3-5x faster analysis
4. **Batch Processing**: Analyze multiple conversations in sequence efficiently
5. **Silent Mode**: Use `print_results=False` for programmatic processing

## 🤝 Contributing

We welcome contributions! Focus areas:
- New conversation analysis metrics
- Integration with popular sales tools
- Performance optimizations
- Additional LLM model support

```bash
git clone https://github.com/DeepMostInnovations/deepmost.git
cd deepmost
pip install -e .[dev]
pytest tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PPO Training**: [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) 
- **LLM Support**: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## 📞 Support & Links

- **Issues**: [GitHub Issues](https://github.com/DeepMostInnovations/deepmost/issues)
- **PyPI**: [https://pypi.org/project/deepmost/](https://pypi.org/project/deepmost/)
- **Models**: [https://huggingface.co/DeepMostInnovations](https://huggingface.co/DeepMostInnovations)
- **Email**: support@deepmost.ai

---

**Focus on what matters: understanding how each conversation turn impacts your sales success.** 🎯

Made with ❤️ by [DeepMost Innovations](https://www.deepmostai.com/)