# run_deepmost_tests.py

import os
import json
from deepmost import prospecting

# --- Main Test: Fast Prospecting & Simulation Workflow ---
print("\n--- Running DeepMost's Fast Prospecting Workflow (Single-Step Search) ---")

prospect_name = "Satya Nadella"
prospect_info = "CEO of Microsoft, AI strategy"

# A capable model is still needed for the agent to understand the search command.
prospecting_model = "unsloth/Qwen3-0.6B"
print(f"Using model_id: {prospecting_model}\n")

try:
    # This single function call now orchestrates the entire fast workflow
    final_plan = prospecting.prospect(
        prospect_name=prospect_name,
        prospect_info=prospect_info,
        model_id=prospecting_model
    )
    
    print("\n--- Final Prospecting Plan Generated ---")
    # Pretty-print the final JSON output
    print(json.dumps(final_plan, indent=2))
    
    if final_plan.get("conversation_plan", {}).get("error"):
        print("\n--- Test Warning: Agent returned a controlled error. ---")
    
    print("\n--- Fast Prospecting Workflow Test Complete ---")

except Exception as e:
    # This will catch unexpected Python errors during the run.
    print(f"\n--- Prospecting Workflow Test Failed with an Exception: {e} ---")
    import traceback
    traceback.print_exc()