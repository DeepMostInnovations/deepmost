# test_conversation_scenarios.py
import logging
from deepmost import sales
import torch # To check device

# --- Configuration ---
LLM_MODEL_FOR_TESTING_METRICS = "unsloth/Llama-3.2-3B-Instruct-GGUF"
# If you have a specific PPO model path, set it here. Otherwise, it uses default.
# PPO_MODEL_PATH = "path/to/your/ppo_model.zip"
PPO_MODEL_PATH = None # Uses default auto-download logic

# Configure basic logging to see DeepMost's internal logs and test script logs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger("test_scenarios")

# --- Test Cases ---
# Each case is a dictionary with 'name', 'conversation', and 'expected_outcome_hint'
CONVERSATION_TEST_CASES = [
    {
        "name": "Strong Positive Interest - Closing",
        "conversation": [
            {"speaker": "customer", "message": "I've reviewed the Enterprise plan details and it has everything we need. The pricing also fits our budget."},
            {"speaker": "sales_rep", "message": "That's fantastic to hear! I'm glad it meets your requirements. Are you ready to move forward with the Enterprise plan today?"},
            {"speaker": "customer", "message": "Yes, absolutely. What are the next steps to get started?"},
            {"speaker": "sales_rep", "message": "Excellent! I can send over the agreement right away and we can schedule your onboarding session. How does that sound?"}
        ],
        "expected_outcome_hint": "very high probability (e.g., > 0.8)"
    },
    {
        "name": "Positive Inquiry - Needs Met",
        "conversation": [
            {"speaker": "customer", "message": "Hi, I'm looking for a CRM that integrates with QuickBooks and has good mobile access for a team of 10."},
            {"speaker": "sales_rep", "message": "Hello! Yes, our Pro plan offers seamless QuickBooks integration and a fully-featured mobile app. It's designed for teams of 5-20 users."},
            {"speaker": "customer", "message": "That sounds perfect. Can you tell me more about the mobile app features?"},
            {"speaker": "sales_rep", "message": "Certainly! The mobile app gives you full access to contacts, deals, tasks, and even allows for offline data syncing. Users love its ease of use on the go."}
        ],
        "expected_outcome_hint": "high probability (e.g., > 0.65)"
    },
    {
        "name": "Neutral Initial Inquiry - Information Gathering",
        "conversation": [
            {"speaker": "customer", "message": "I'm just starting to look for a CRM. What makes your product different?"},
            {"speaker": "sales_rep", "message": "Thanks for asking! We focus on a user-friendly interface, powerful automation tools to save you time, and excellent customer support. What are some of the key challenges you're facing with your current sales process?"},
            {"speaker": "customer", "message": "Mainly disorganization and follow-up reminders."},
             {"speaker": "sales_rep", "message": "Our CRM can definitely help with that by automating reminders and providing a clear view of all your customer interactions."}
        ],
        "expected_outcome_hint": "medium probability (e.g., 0.4 - 0.6)"
    },
    {
        "name": "Price Objection - Unresolved",
        "conversation": [
            {"speaker": "customer", "message": "Your product looks good, but the Pro plan at $49/user/month is much more expensive than Competitor X."},
            {"speaker": "sales_rep", "message": "I understand your concern about pricing. While Competitor X might seem cheaper upfront, our Pro plan includes [Feature A] and [Feature B] which are often add-ons with them. Plus, our users report a 20% increase in efficiency. Could the long-term value outweigh the initial cost difference for you?"},
            {"speaker": "customer", "message": "I'm not sure. That's still a big jump for us right now. I need to think about it."},
            {"speaker": "sales_rep", "message": "I respect that. Would it be helpful to explore if our Basic plan could meet some of your immediate needs at a lower price point, or perhaps discuss a possible annual discount for the Pro plan?"}
        ],
        "expected_outcome_hint": "low probability (e.g., < 0.4)"
    },
    {
        "name": "Clear Mismatch - Needs Not Met",
        "conversation": [
            {"speaker": "customer", "message": "I need a completely free CRM for just myself, and I also need it to manage my crypto portfolio."},
            {"speaker": "sales_rep", "message": "Thank you for reaching out. Currently, all our plans are subscription-based, designed for team collaboration in sales, and we don't offer features for cryptocurrency portfolio management. It sounds like our CRM might not be the best fit for your specific needs at this time."},
            {"speaker": "customer", "message": "Oh, okay. Thanks anyway."},
        ],
        "expected_outcome_hint": "very low probability (e.g., < 0.15)"
    },
    {
        "name": "Customer Disengagement - Short Interaction",
        "conversation": [
            {"speaker": "customer", "message": "Tell me about your CRM."},
            {"speaker": "sales_rep", "message": "We offer a comprehensive CRM solution! What are you looking for specifically?"},
            {"speaker": "customer", "message": "Just browsing."},
            {"speaker": "sales_rep", "message": "Okay, well feel free to check out our website for feature details or let me know if specific questions come up!"}
        ],
        "expected_outcome_hint": "low to very low probability (e.g., < 0.3)"
    }
]

def run_scenario_tests():
    logger.info("Initializing DeepMost Agent for scenario testing...")
    logger.info(f"Using LLM for metrics: {LLM_MODEL_FOR_TESTING_METRICS}")
    if PPO_MODEL_PATH:
        logger.info(f"Using PPO model from path: {PPO_MODEL_PATH}")

    use_gpu_flag = torch.cuda.is_available()
    logger.info(f"GPU available: {use_gpu_flag}")

    try:
        agent = sales.Agent(
            llm_model=LLM_MODEL_FOR_TESTING_METRICS,
            model_path=PPO_MODEL_PATH, # Will use default if None
            use_gpu=use_gpu_flag
        )
        logger.info("Agent initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize DeepMost Agent: {e}", exc_info=True)
        logger.error("Aborting tests as Agent could not be initialized.")
        return

    logger.info("\n" + "="*10 + " Starting Conversation Scenario Tests " + "="*10)

    for i, case in enumerate(CONVERSATION_TEST_CASES):
        logger.info(f"\n--- Test Case {i+1}: {case['name']} ---")
        logger.info(f"Expected Outcome Hint: {case['expected_outcome_hint']}")
        logger.info("Conversation:")
        for turn in case['conversation']:
            logger.info(f"  {turn['speaker'].capitalize()}: {turn['message']}")

        try:
            # For each scenario, predict on the *full* conversation provided
            # The agent internally handles turn progression if you were to feed it message by message,
            # but for a full scenario test, we give it the whole history.
            # The Agent's predict method will reset its internal turn counter for a new conversation_id
            # if one is not explicitly managed and passed, or if it's a new list object.
            # To be safe, we can assume predict processes the given list as one complete observation.
            result = agent.predict(case['conversation']) # Agent manages its own state across calls if same conv_id were used
                                                        # but here each call to predict is for a new, distinct conversation scenario.

            probability = result['probability']
            status = result['status']
            metrics = result['metrics']

            logger.info(f"Predicted Conversion Probability: {probability:.2%}")
            logger.info(f"Status: {status}")
            logger.info(f"Metrics: {metrics}")
            logger.info(f"--- Test Case '{case['name']}' Finished ---")

        except Exception as e:
            logger.error(f"Error during prediction for case '{case['name']}': {e}", exc_info=True)
            logger.info(f"--- Test Case '{case['name']}' FAILED due to error ---")

    logger.info("\n" + "="*10 + " Conversation Scenario Tests Finished " + "="*10)

if __name__ == "__main__":
    run_scenario_tests()