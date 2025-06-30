"""
High-level API for prospect analysis and sales strategy generation.
"""
import json
import logging
from typing import Dict, Any, Optional

# Smol Agents for web research
from smolagents import CodeAgent, DuckDuckGoSearchTool, VLLMModel

# DeepMost for conversation analysis
from .sales import Agent as SalesAgent

logger = logging.getLogger(__name__)

class Profiler:
    """
    A sales prospecting agent that researches a person and generates a
    high-conversion conversation strategy using DeepMost.
    This implementation uses a VLLM backend for high-performance inference.
    """
    def __init__(
        self,
        vllm_model_id: str = "HuggingFaceH4/zephyr-7b-beta",
        deepmost_sales_agent: Optional[SalesAgent] = None
    ):
        """
        Initializes the Profiler with a VLLM backend.

        Args:
            vllm_model_id: The Hugging Face model identifier for the model
                           to be used for web research analysis via VLLM.
            deepmost_sales_agent: An initialized instance of deepmost.sales.Agent.
                                  If None, a default open-source agent will be created.
        """
        logger.info(f"Initializing web research agent with VLLM model: {vllm_model_id}")
        self.research_model = VLLMModel(model_id=vllm_model_id)
        self.research_agent = CodeAgent(
            tools=[DuckDuckGoSearchTool()], 
            model=self.research_model
        )

        if deepmost_sales_agent:
            self.sales_agent = deepmost_sales_agent
        else:
            logger.info("No sales agent provided, initializing default open-source DeepMost agent.")
            self.sales_agent = SalesAgent()

    def generate_profile_and_strategy(
        self,
        person_name: str,
        company: Optional[str] = None,
        title: Optional[str] = None,
        additional_info: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive profile and a high-conversion conversation strategy.

        Args:
            person_name: The full name of the person to research.
            company: The company they work for.
            title: Their job title.
            additional_info: Any other relevant information to guide the search.

        Returns:
            A dictionary containing the generated profile and conversation strategy.
        """
        # Step 1: Build the research prompt for the smol-agent
        search_query = f"{person_name}"
        if title:
            search_query += f", {title}"
        if company:
            search_query += f" at {company}"

        prompt = f"""
        You are an expert sales intelligence analyst. Your task is to research the person "{search_query}"
        and compile a comprehensive professional profile.

        Search the web for the following information:
        1.  **Professional Background**: LinkedIn profile, career history, key achievements, and areas of expertise.
        2.  **Current Role**: Responsibilities at their current company, recent projects.
        3.  **Company Information**: What their company does, recent news, funding rounds, or product launches.
        4.  **Potential Interests**: Look for personal blogs, Twitter/X activity, conference talks, or interviews to identify potential hobbies or professional interests (e.g., specific technologies, management styles, etc.).
        5.  **Location**: General location (city/state).

        CRITICAL: Consolidate your findings into a single, valid JSON object.
        The JSON object must have the following keys: "name", "company", "title", "location",
        "professional_summary", "company_summary", "potential_talking_points".
        "potential_talking_points" should be a list of 3-5 actionable conversation starters based on your research.

        Now, research "{search_query}" and provide the JSON output.
        """
        
        # Step 2: Run the research agent
        logger.info(f"Starting web research for: {search_query}")
        try:
            research_output = self.research_agent.run(prompt)
            json_str = research_output[research_output.find('{'):research_output.rfind('}')+1]
            profile = json.loads(json_str)
            logger.info("Successfully generated prospect profile.")
        except Exception as e:
            logger.error(f"Failed to generate profile for {person_name}: {e}")
            return {"error": "Failed to generate profile.", "details": str(e)}

        # Step 3: Generate and analyze conversation strategy
        logger.info("Generating and analyzing conversation strategy with DeepMost.")
        system_prompt = f"""
        You are a top-performing sales representative. You are about to engage with {profile.get('name')},
        the {profile.get('title')} at {profile.get('company')}.
        Based on your research, you know:
        - Summary: {profile.get('professional_summary')}
        - Talking Points: {', '.join(profile.get('potential_talking_points', []))}
        Your goal is to initiate a conversation that builds rapport and has a high probability of leading to a sale.
        Use one of the talking points to craft your opening message.
        """
        response_result = self.sales_agent.predict_with_response(
            conversation=[], user_input="<Begin Conversation>", system_prompt=system_prompt
        )
        sales_rep_opener = response_result['response']
        
        customer_system_prompt = f"""
        You are {profile.get('name')}. You just received the message: "{sales_rep_opener}".
        Given your background, write a brief, plausible response.
        """
        customer_response_result = self.sales_agent.predict_with_response(
            conversation=[{'speaker': 'sales_rep', 'message': sales_rep_opener}],
            user_input="<Generate Customer Response>", system_prompt=customer_system_prompt
        )
        customer_first_reply = customer_response_result['response']

        # Step 4: Analyze and return
        simulated_conversation = [
            {"speaker": "sales_rep", "message": sales_rep_opener},
            {"speaker": "customer", "message": customer_first_reply}
        ]
        analysis = self.sales_agent.analyze_conversation_progression(simulated_conversation, print_results=False)
        final_prediction = analysis[-1]

        return {
            "prospect_profile": profile,
            "conversation_strategy": {
                "simulated_conversation": simulated_conversation,
                "final_conversion_probability": final_prediction['probability'],
                "status": final_prediction['status'],
                "suggested_action": final_prediction['metrics'].get('suggested_action', 'Continue building value.'),
            },
            "backend_used": {
                "profiling_agent": f"vllm::{vllm_model_id}",
                "sales_agent": self.sales_agent.backend_type
            }
        }