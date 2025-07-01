# deepmost/prospecting.py (With a more robust prompt)

import json
from typing import Dict, Any, List

try:
    from smolagents import CodeAgent, Tool, TransformersModel, WebSearchTool
except ImportError:
    raise ImportError("smolagents is not installed. Please install it with `pip install deepmost[prospecting]` or `pip install smolagents[toolkit]`")

from .sales import Agent as SalesAgent

class ProfileBuilderTool(Tool):
    """Tool to build a structured JSON profile from unstructured text."""
    name = "profile_builder"
    description = "Compiles unstructured text about a person into a structured JSON profile."
    inputs = {
        "person_name": {"type": "string", "description": "The full name of the person."},
        "information": {"type": "string", "description": "Unstructured text containing information about the person from web search."},
    }
    output_type = "string"

    def forward(self, person_name: str, information: str) -> str:
        """Creates a detailed JSON profile."""
        profile = {
            "name": person_name,
            "summary": f"Profile based on web search for {person_name}.",
            "unstructured_data": information,
            "potential_interests": ["AI-driven efficiency", "CRM solutions", "Sales technology"],
            "pain_points_hypothesis": ["Inefficient lead prioritization", "Poor sales forecasting", "Manual follow-up processes"],
            "company_name": "Microsoft" 
        }
        return json.dumps(profile, indent=2)

class RealTimeSalesSimulatorTool(Tool):
    """
    Tool to simulate a turn in a sales conversation for real-time assistance.
    It takes the conversation history and generates the next turn.
    """
    name = "real_time_sales_simulator"
    description = "Simulates the next turn of a sales conversation to generate a high-conversion dialogue."
    inputs = {
        "prospect_profile_json": {"type": "string", "description": "The JSON profile of the prospect."},
        "conversation_history": {"type": "array", "description": "The list of previous messages in the conversation."},
        "salesperson_message": {"type": "string", "description": "The latest message from the salesperson to the prospect."},
    }
    output_type = "string"

    def forward(self, prospect_profile_json: str, conversation_history: list, salesperson_message: str) -> str:
        """
        Uses the deepmost.sales.Agent to simulate the next turn of the conversation.
        """
        try:
            profile = json.loads(prospect_profile_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON profile provided."})

        system_prompt = f"""
        You are a helpful sales assistant. Your goal is to generate a response from the prospect, '{profile.get('name', 'the client')}',
        that continues the conversation in a realistic way. The prospect is interested in solving challenges like
        {profile.get('pain_points_hypothesis', [])}. Your response should reflect this context.
        """
        
        sales_agent = sales.Agent() 
        result = sales_agent.predict_with_response(
            conversation=conversation_history,
            user_input=salesperson_message,
            system_prompt=system_prompt
        )
        
        return json.dumps(result, indent=2)

class ProspectingAgent:
    """
    An agent that researches a prospect and generates a real-time, high-conversion
    conversation plan.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct", use_gpu: bool = True):
        """Initializes the ProspectingAgent."""
        self.model = TransformersModel(
            model_id=model_id,
            max_new_tokens=4096,
            device_map="auto"
        )
        self.agent = CodeAgent(
            tools=[WebSearchTool(), ProfileBuilderTool(), RealTimeSalesSimulatorTool()],
            model=self.model,
        )

    def generate_plan(self, prospect_name: str, prospect_info: str) -> Dict[str, Any]:
        """
        Generates the initial profile and the first turn of the conversation.
        """
        # IMPROVED PROMPT for better tool orchestration
        prompt = f"""
        Your task is to generate a sales plan for '{prospect_name}' ({prospect_info}).
        You MUST call the tools in this exact order to build the plan step-by-step.

        Step 1: Call the `web_search` tool to find professional information on '{prospect_name}'.
        Step 2: Take the output from Step 1 and use it as the 'information' argument for the `profile_builder` tool.
        Step 3: Devise a compelling opening sales message based on the created profile.
        Step 4: Call the `real_time_sales_simulator` tool. Use the JSON profile from Step 2, an empty list `[]` for 'conversation_history', and your devised opening message for 'salesperson_message'.

        The final output of your execution MUST be the JSON string from Step 4. Do not add any other text.
        """
        final_result_str = self.agent.run(prompt)
        
        try:
            return json.loads(final_result_str)
        except (json.JSONDecodeError, TypeError):
            return {
                "error": "The agent did not produce a valid JSON output.",
                "raw_output": final_result_str
            }

def prospect(prospect_name: str, prospect_info: str, **kwargs) -> Dict[str, Any]:
    """
    High-level function to generate an initial prospecting plan and conversation starter.
    """
    agent = ProspectingAgent(**kwargs)
    return agent.generate_plan(prospect_name, prospect_info)