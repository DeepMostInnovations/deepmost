# run_deepmost_tests.py

import os
import json
from deepmost import prospecting

# --- Main Test: Fast Prospecting & Simulation Workflow ---
print("\n--- Running DeepMost's Fast Prospecting Workflow ---")

prospect_name = "Satya Nadella"
prospect_info = "CEO of Microsoft, AI strategy"

# Define the two different models to be used
search_model = "unsloth/Qwen3-0.6B"
simulation_model = "unsloth/Qwen3-4B-GGUF"

print(f"Using Search Model (safetensors): {search_model}")
print(f"Using Simulation Model (GGUF): {simulation_model}\n")

try:
    # Pass the specific models to the prospect function
    final_plan = prospecting.prospect(
        prospect_name=prospect_name,
        prospect_info=prospect_info,
        search_model_id=search_model,
        simulation_model_id=simulation_model
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