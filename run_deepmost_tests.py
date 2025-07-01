# run_deepmost_tests.py (Using a more capable model for prospecting)

import os
import json
from deepmost import sales, prospecting

# --- Test 1: Open-Source Sales Module ---
print("--- Running DeepMost Sales Module Test (Open-Source) ---")
print("Skipping standalone sales module test to focus on prospecting workflow.\n")

# --- Test 2: Open-Source Prospecting & Real-time Assistance Workflow ---
print("\n--- Running DeepMost Prospecting & Assistance Test (Open-Source) ---")
prospect_name = "Satya Nadella"
prospect_info = "CEO of Microsoft, focusing on AI transformation"

# SWITCHING to a more capable model that can handle code generation and tool orchestration.
prospecting_model = "unsloth/Qwen3-0.6B"
print(f"Testing prospecting.prospect with model_id: {prospecting_model}\n")

try:
    # Step 1: Generate the initial plan and conversation starter
    initial_plan = prospecting.prospect(
        prospect_name=prospect_name,
        prospect_info=prospect_info,
        model_id=prospecting_model
    )
    print("--- Initial Prospecting Plan Generated ---")
    print(json.dumps(initial_plan, indent=2))
    
    print("\n--- Simulating Real-time Assistance (Next Turn) ---")
    
    if 'prediction' in initial_plan:
        conversation_history = [
            {"speaker": "sales_rep", "message": "Hi Satya, given Microsoft's focus on AI, I wanted to share how our AI-CRM is helping leaders boost sales efficiency."},
            {"speaker": "customer", "message": initial_plan['response']}
        ]
        
        next_sales_message = "I'm glad it sounds interesting. Many leaders are finding value in our automated lead prioritization. Would a brief 15-minute demo be useful to see it in action?"

        sales_agent = sales.Agent(llm_model=prospecting_model) # Reuse the same model
        next_turn_result = sales_agent.predict_with_response(
            conversation=conversation_history,
            user_input=next_sales_message,
            system_prompt="You are Satya Nadella. Respond to the salesperson's message about a demo."
        )

        print("--- Next Turn Simulation Result ---")
        print(json.dumps(next_turn_result, indent=2))
    
    print("\n--- Open-Source Prospecting & Assistance Test Complete ---")

except Exception as e:
    print(f"\n--- Open-Source Prospecting & Assistance Test Failed: {e} ---")