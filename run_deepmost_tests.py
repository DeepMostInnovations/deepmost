import json
import logging
from deepmost.prospecting import Profiler

# --- Optional: Configure logging to see more details ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_local_test():
    """
    A simple function to test the local, editable installation of the Profiler.
    """
    # --- 1. Initialize the Profiler ---
    # Because you are in an editable install, this uses the code directly
    # from your local 'deepmost/prospecting.py' file.
    logging.info("Initializing the DeepMost Profiler with VLLM backend...")
    
    try:
        profiler = Profiler(
            # Using a relatively small model is good for testing
            vllm_model_id="HuggingFaceTB/SmolLM-135M-Instruct"
        )
    except Exception as e:
        logging.error(f"Failed to initialize Profiler. Ensure you have a compatible NVIDIA GPU and drivers. Error: {e}")
        return

    logging.info("Profiler initialized successfully.")

    # --- 2. Define Your Prospect ---
    prospect_name = "Elon Musk"
    prospect_company = "Tesla"
    prospect_title = "CEO"

    # --- 3. Generate the Profile and Strategy ---
    logging.info(f"Generating profile and strategy for {prospect_name}...")
    analysis_result = profiler.generate_profile_and_strategy(
        person_name=prospect_name,
        company=prospect_company,
        title=prospect_title
    )

    # --- 4. Review the Results ---
    print("\n" + "="*25 + " ANALYSIS COMPLETE " + "="*25)
    print(json.dumps(analysis_result, indent=2))
    print("="*75 + "\n")

    if 'error' in analysis_result:
        logging.error("The analysis failed. Please check the error details above.")
    else:
        logging.info("Analysis completed successfully.")

if __name__ == "__main__":
    run_local_test()