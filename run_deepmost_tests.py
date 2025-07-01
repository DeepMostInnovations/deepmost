# run_deepmost_tests.py (Simplified and focused)

import os
import json
from deepmost import sales, prospecting

# --- Main Test: Open-Source Prospecting & Real-time Assistance Workflow ---
print("\n--- Running DeepMost Prospecting & Assistance Test (Open-Source) ---")

prospect_name = "Satya Nadella"
prospect_info = "CEO of Microsoft, focusing on AI transformation"

# A capable model is CRITICAL for the agent to correctly orchestrate tool calls.
# Using a smaller model will likely result in the agent getting stuck.
prospecting_model = "unsloth/Qwen3-0.6B"
print(f"Testing prospecting.prospect with model_id: {prospecting_model}\n")

try:
    # This function now uses a more robust prompt to guide the agent.
    initial_plan = prospecting.prospect(
        prospect_name=prospect_name,
        prospect_info=prospect_info,
        model_id=prospecting_model
    )
    
    print("--- Initial Prospecting Plan Generated ---")
    # Pretty-print the JSON output
    print(json.dumps(initial_plan, indent=2))
    
    if initial_plan.get("error"):
        print("\n--- Test Warning: Agent returned a controlled error. ---")
    
    print("\n--- Open-Source Prospecting & Assistance Test Complete ---")

except Exception as e:
    # This will catch unexpected Python errors during the run.
    print(f"\n--- Open-Source Prospecting & Assistance Test Failed with an Exception: {e} ---")
    import traceback
    traceback.print_exc()