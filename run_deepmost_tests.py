from deepmost import sales

conversation = [
    "Hello, I'm looking for information on your new AI-powered CRM",
    "You've come to the right place! Our new AI CRM is designed to help businesses like yours increase sales efficiency. What specific challenges are you facing?",
    "not interested, bye",
    "Excellent, those are two key strengths. Our AI analyzes lead behavior and automatically suggests optimal follow-up times. Would you like to see a demo?",
    "That sounds interesting. What's the pricing like?"
]

# Get turn-by-turn analysis with automatic output
results = sales.analyze_progression(conversation, llm_model="unsloth/Qwen3-4B-GGUF")