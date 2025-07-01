# deepmost/prospecting.py

import json
from typing import Dict, Any

try:
    from smolagents import CodeAgent, TransformersModel, WebSearchTool
except ImportError:
    raise ImportError("smolagents is not installed. Please install it with `pip install deepmost[prospecting]` or `pip install smolagents[toolkit]`")

from .sales import Agent as SalesAgent

# --- Tool-like classes, now used as regular Python objects ---

class ProfileBuilder:
    """Class to build a structured JSON profile from unstructured text."""
    def build(self, person_name: str, information: str) -> Dict[str, Any]:
        """Creates a detailed JSON profile from search results."""
        profile = {
            "name": person_name,
            "summary": f"Profile based on web search for {person_name}.",
            "unstructured_data": information,
            "potential_interests": ["AI-driven efficiency", "CRM solutions", "Sales technology"],
            "pain_points_hypothesis": ["Inefficient lead prioritization", "Poor sales forecasting", "Manual follow-up processes"],
            "company_name": "Microsoft"  # This could be extracted from info in a real scenario
        }
        return profile

class RealTimeSalesSimulator:
    """Class to simulate the first turn of a sales conversation."""
    def simulate_first_turn(self, prospect_profile: Dict[str, Any], llm_model: str) -> Dict[str, Any]:
        """
        Uses deepmost.sales.Agent to generate an opening and simulate a response.
        """
        system_prompt = f"""
        You are a helpful sales assistant. Your goal is to generate a realistic response from the prospect, '{prospect_profile.get('name', 'the client')}',
        who is interested in solving challenges like {prospect_profile.get('pain_points_hypothesis', [])}.
        Your response should reflect this context.
        """
        
        # Devise a compelling opening line based on the profile
        opening_message = (
            f"Hi {prospect_profile.get('name', 'there')}, I saw you're interested in '{prospect_profile.get('potential_interests', ['AI'])[0]}'. "
            f"Given your focus at {prospect_profile.get('company_name', 'your company')}, I thought you'd find our AI-CRM's approach "
            f"to solving {prospect_profile.get('pain_points_hypothesis', ['key challenges'])[0]} interesting."
        )

        sales_agent = SalesAgent(llm_model=llm_model)
        result = sales_agent.predict_with_response(
            conversation=[],  # Start with an empty conversation
            user_input=opening_message,
            system_prompt=system_prompt
        )
        
        # Add the opening message to the result for clarity
        result['opening_message'] = opening_message
        return result

# --- Agent for Single-Step Web Search ---

class SearchAgent:
    """A simplified agent whose only job is to perform a web search."""
    def __init__(self, model_id: str, use_gpu: bool = True):
        self.model = TransformersModel(
            model_id=model_id,
            max_new_tokens=2048,  # Can be smaller as we only need search results
            device_map="auto"
        )
        self.agent = CodeAgent(
            tools=[WebSearchTool()],
            model=self.model,
            max_steps=1
        )

    def search(self, query: str) -> str:
        """Runs a web search and returns the summarized results as a string."""
        prompt = f"Please perform a web search for the following query and return the summarized results: '{query}'"
        search_results = self.agent.run(prompt)
        return search_results

# --- High-Level Orchestrator Function ---

def plan_and_simulate(prospect_name: str, prospect_info: str, model_id: str) -> Dict[str, Any]:
    """
    Orchestrates the fast, single-step search and subsequent processing.
    """
    print("Step 1: Using LLM Agent for single-step web search...")
    search_agent = SearchAgent(model_id=model_id)
    search_query = f"{prospect_name}, {prospect_info}"
    search_results = search_agent.search(search_query)
    print("...Web search complete.")

    print("Step 2: Building profile with deterministic Python code...")
    profile_builder = ProfileBuilder()
    profile = profile_builder.build(prospect_name, search_results)
    print("...Profile built.")
    
    print("Step 3: Simulating conversation with deterministic Python code...")
    simulator = RealTimeSalesSimulator()
    simulation_result = simulator.simulate_first_turn(profile, model_id)
    print("...Simulation complete.")
    
    final_plan = {
        "prospect_profile": profile,
        "conversation_plan": simulation_result
    }
    
    return final_plan

def prospect(prospect_name: str, prospect_info: str, **kwargs) -> Dict[str, Any]:
    """
    High-level function to generate an initial prospecting plan and conversation starter
    using the fast, single-step search workflow.
    """
    model_id = kwargs.get("model_id", "unsloth/Qwen3-4B-GGUF")
    return plan_and_simulate(prospect_name, prospect_info, model_id)