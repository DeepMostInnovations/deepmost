# deepmost/prospecting.py

import json
from typing import Dict, Any

try:
    from smolagents import CodeAgent, TransformersModel, WebSearchTool
except ImportError:
    raise ImportError("smolagents is not installed. Please install it with `pip install deepmost[prospecting]` or `pip install smolagents[toolkit]`")

from .sales import Agent as SalesAgent

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
            "company_name": "Microsoft"
        }
        return profile

class RealTimeSalesSimulator:
    """Class to simulate the first turn of a sales conversation."""
    def simulate_first_turn(self, prospect_profile: Dict[str, Any], gguf_model_id: str) -> Dict[str, Any]:
        """
        Uses deepmost.sales.Agent with a GGUF model to simulate a response.
        """
        system_prompt = f"""
        You are a helpful sales assistant. Your goal is to generate a realistic response from the prospect, '{prospect_profile.get('name', 'the client')}',
        who is interested in solving challenges like {prospect_profile.get('pain_points_hypothesis', [])}.
        Your response should reflect this context.
        """
        
        opening_message = (
            f"Hi {prospect_profile.get('name', 'there')}, I saw you're interested in '{prospect_profile.get('potential_interests', ['AI'])[0]}'. "
            f"Given your focus at {prospect_profile.get('company_name', 'your company')}, I thought you'd find our AI-CRM's approach "
            f"to solving {prospect_profile.get('pain_points_hypothesis', ['key challenges'])[0]} interesting."
        )

        # This part requires a GGUF model because of how sales.Agent is built
        sales_agent = SalesAgent(llm_model=gguf_model_id)
        result = sales_agent.predict_with_response(
            conversation=[],
            user_input=opening_message,
            system_prompt=system_prompt
        )
        
        result['opening_message'] = opening_message
        return result

class SearchAgent:
    """A simplified agent whose only job is to perform a web search."""
    def __init__(self, model_id: str, use_gpu: bool = True):
        self.model = TransformersModel(
            model_id=model_id,
            max_new_tokens=2048,
            device_map="auto"
        )
        self.agent = CodeAgent(
            tools=[WebSearchTool()],
            model=self.model,
            max_steps=1
        )

    def search(self, query: str) -> str:
        """Runs a web search and returns the summarized results."""
        prompt = f"Please perform a web search for the following query and return the summarized results: '{query}'"
        search_results = self.agent.run(prompt)
        return search_results

def plan_and_simulate(
    prospect_name: str,
    prospect_info: str,
    search_model_id: str,
    simulation_model_id: str
) -> Dict[str, Any]:
    """
    Orchestrates the fast, single-step search and subsequent processing.
    """
    print(f"Step 1: Using '{search_model_id}' for single-step web search...")
    search_agent = SearchAgent(model_id=search_model_id)
    search_query = f"{prospect_name}, {prospect_info}"
    search_results = search_agent.search(search_query)
    print("...Web search complete.")

    print("Step 2: Building profile with deterministic Python code...")
    profile_builder = ProfileBuilder()
    profile = profile_builder.build(prospect_name, search_results)
    print("...Profile built.")
    
    print(f"Step 3: Simulating conversation with GGUF model '{simulation_model_id}'...")
    simulator = RealTimeSalesSimulator()
    simulation_result = simulator.simulate_first_turn(profile, simulation_model_id)
    print("...Simulation complete.")
    
    final_plan = {
        "prospect_profile": profile,
        "conversation_plan": simulation_result
    }
    
    return final_plan

def prospect(
    prospect_name: str,
    prospect_info: str,
    search_model_id: str = "unsloth/Qwen3-0.6B",
    simulation_model_id: str = "unsloth/Qwen3-4B-GGUF",
    **kwargs
) -> Dict[str, Any]:
    """
    High-level function to generate an initial prospecting plan.
    - search_model_id: A standard Hugging Face model for the search agent.
    - simulation_model_id: A GGUF-compatible model for conversation simulation.
    """
    return plan_and_simulate(prospect_name, prospect_info, search_model_id, simulation_model_id)