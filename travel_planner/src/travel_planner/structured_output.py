from agents import (
    Agent, Runner, function_tool, input_guardrail, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered, RunContextWrapper, TResponseInputItem
    )
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@function_tool
def get_weather(destination: str):
    """Get the weather of the destination"""
    if destination == "Rotterdam":
        return f"The weather in Rotterdam is hot with a temperature of 30 degrees Celsius."
    elif destination == "Paris":
        return f"The weather in Paris is pleasant with a temperature of 20 degrees Celsius."
    else:
        return f"The weather in {destination} is unknown."
    
@function_tool
def get_flight(destination:str):
    """Get the flight information for the destination"""
    if destination == "Rotterdam":
        return f"The latest flight to Rotterdam is on Monday at 10:00 AM. Airline: KLM, Flight Number: KL1234"
    elif destination == "Paris":
        return f"The latest flight to Paris is on Tuesday at 2:00 PM. Airline: Air France, Flight Number: AF1234"
    else:
        return f"The flight to {destination} is unknown."


class TravelPlan(BaseModel):
    destination: str = Field(description="The destination of the travel plan")
    duration_days: int = Field(description="The duration of the travel plan in days")
    budget: int = Field(description="The budget of the travel plan in USD")
    activities: list[str] = Field(description="The activities to do in the destination")
    weather: str = Field(description="The weather of the destination")
    notes: str = Field(description="Any additional notes or recommendations about the travel plan")

class FlightPlan(BaseModel):
    destination: str = Field(description="The destination of the travel plan")
    flight_date: str = Field(description="The date of the flight")
    flight_number: str = Field(description="The flight number")
    airline: str = Field(description="The airline")

class BudgetAnalysis(BaseModel):
    is_realistic: bool = Field(description="Whether the budget is realistic for the destination")
    reasoning: str = Field(description="The reasoning for the budget analysis")
    suggested_budget: int = Field(description="The suggested budget for the destination")

budget_analysis_agent = Agent(
    name="Budget Analysis Agent",
    instructions="""You analyze travel budget to determine if it is realistic for the destination and duration of the trip.
    Consider factors like:
    - Average hotel costs in the destination
    - Food and Entertainment costs
    - Flight costs
    - Local transportation costs
    - Local attractions and activities costs

    Provide a clear analysis of whether the budget is realistic and why.
    If it is not realistic, suggest a more realistic budget.
    If no budget was mentioned, just assume it is realistic.
    """,
    output_type=BudgetAnalysis,
)

@input_guardrail
async def budget_guardrail(ctx: RunContextWrapper[None], agent: Agent,
                            input: str|list[TResponseInputItem]) -> GuardrailFunctionOutput:
    """Check if the user's budget is realistic for the destination and duration of the trip."""
    try:
        analysis_prompt = f""" User is planning a trip and said {input}.
        Analyze if their budget is realistic for a trip to their destination for the mentioned duration.
        """
        result = await Runner.run(budget_analysis_agent, input=analysis_prompt, context=ctx.context)
        output_info = result.final_output_as(BudgetAnalysis)
        if not output_info.is_realistic:
            print(f"Budget is not realistic for the trip. {output_info.reasoning}" if not output_info.is_realistic else None)
        return GuardrailFunctionOutput(
            output_info=output_info,
            tripwire_triggered=not output_info.is_realistic,
            )
    except Exception as e:
        print(f"Error in budget guardrail: {e}")
        return GuardrailFunctionOutput(
            tripwire_triggered=False,
            output_info=BudgetAnalysis(
                is_realistic=True,
                reasoning=f"Error in budget analysis: {str(e)}",
                suggested_budget=0,
            ),
        )
       
flight_agent = Agent(
    name="Flight Agent",
    instructions="""You are a flight agent that help users plan their perfect trip.
    You can create personalized flight plans based on the user's preferences and interest.
    Provide specific recommendations based on user preferences and interests.
    """,
    handoff_description="""Provide information about flights only,
      including departure times, departure dates, airlines, and flight numbers.
    """,
    tools=[get_flight],
    output_type=FlightPlan,
)

travel_agent = Agent(
    name="Travel Agent",
    instructions="""You are a travel plannig agent that help users plan their perfect trip.
    You can create personalized travel plans based on the user's preferences and interest.
    Provide specific recommendations based on user preferences and interests.
    When creating travel plans, consider the following:
    - Duration of the trip
    - Budget
    - Local attarctions and activities
    """,
    handoffs=[flight_agent],
    tools=[get_weather],
    input_guardrails=[budget_guardrail],
    output_type=TravelPlan,
)


def main():
    queries = [
        "I want to travel to Paris for 5 days. My budget is $10000. How is the weather there? What should I do there?",
        "How soon can I fly to Rotterdam?",       
        "I want to travel to Dubai for 10 days. My budget is $20000. What should I do there?",
    ]

    for query in queries:
        print(f"Query: {query}")
        result = Runner.run_sync(travel_agent, input=query)
        
        # Check which agent was last used
        last_agent_name = result.last_agent.name
        print(f"Last agent: {last_agent_name}")
        
        # Get and display the appropriate structured output based on agent
        if last_agent_name == "Travel Agent":
            output = result.final_output_as(TravelPlan)
            print(output)
      
        elif last_agent_name == "Flight Agent":
            output = result.final_output_as(FlightPlan)
            print(output)
            
        print("-"*100)

if __name__ == "__main__":
    main()


