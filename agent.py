import os
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. DEFINE THE AGENT STATE
# ==========================================
class MealPrepState(TypedDict):
    # Inputs loaded from your data files
    allergies: str
    inventory: str
    diet_history: str
    
    # Context fetched by tools
    location: str
    weather: str
    
    # Final AI Output
    meal_plan: str

# ==========================================
# 2. DEFINE THE SMART HOME/CONTEXT TOOLS
# ==========================================
@tool
def get_current_location() -> str:
    """Fetches the user's current city based on network location."""
    # Mocking location for the cloud agent environment
    return "London, UK"

@tool
def get_weather_forecast(location: str) -> str:
    """Gets the current weather conditions for a given location to adapt meal temperatures."""
    # Mocking a rainy day context as requested in your requirements
    return "It is currently 12°C and pouring rain outside."

# Package tools so the graph can execute them if needed
tools = [get_current_location, get_weather_forecast]
tool_node = ToolNode(tools)

# ==========================================
# 3. DEFINE THE GRAPH NODES (The Brains)
# ==========================================

def load_data_files(state: MealPrepState) -> MealPrepState:
    """Node 1: Reads your text files straight from GitHub into the graph state."""
    try:
        with open("./data/allergies.txt", "r") as f:
            allergies = f.read()
        with open("./data/grocery_receipt.txt", "r") as f:
            inventory = f.read()
        with open("./data/diet_history.txt", "r") as f:
            diet_history = f.read()
    except Exception as e:
        allergies = "None provided."
        inventory = "Eggs, Chicken, Spinach."
        diet_history = "No history available."
        print(f"Error loading files, using fallback defaults: {e}")

    return {
        "allergies": allergies,
        "inventory": inventory,
        "diet_history": diet_history
    }

def fetch_environment_context(state: MealPrepState) -> MealPrepState:
    """Node 2: Runs our location and weather tools to get real-time context."""
    # We call our mock tools directly inside the python workflow
    loc = get_current_location.invoke({})
    weather = get_weather_forecast.invoke({"location": loc})
    
    return {
        "location": loc,
        "weather": weather
    }

def generate_meal_plan(state: MealPrepState) -> MealPrepState:
    """Node 3: Combines data files + weather tools to generate the final recipe plan."""
    # Initialize our LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Construct the ultimate contextual system prompt
    system_prompt = f"""
    You are a Personal AI Health & Meal Prep Coach. Your task is to generate a custom 1-day meal plan (Breakfast, Lunch, Dinner).
    
    You MUST strictly align your recommendations with the following contextual constraints:
    
    [USER ALLERGIES & DIET LAWS]
    {state['allergies']}
    
    [AVAILABLE INGREDIENTS IN FRIDGE]
    {state['inventory']}
    
    [PAST MEAL HISTORY & PREFERENCES]
    {state['diet_history']}
    
    [CURRENT ENVIRONMENT WEATHER]
    {state['weather']} (Location: {state['location']})
    
    Instructions:
    1. Do not use ingredients the user is allergic to.
    2. Adjust the meal temperature vibe to the weather (e.g., if raining/cold, suggest comforting warm foods).
    3. Look at history: if they hated a meal or preparation style previously, adjust your recipe style!
    4. Provide an estimated preparation/cooking time for each meal.
    """
    
    user_prompt = "Generate my meal plan for today based on what I have in my fridge right now."
    
    # Invoke OpenAI
    response = llm.invoke([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    
    return {"meal_plan": response.content}

# ==========================================
# 4. BUILD AND COMPILE THE LANGGRAPH
# ==========================================
builder = StateGraph(MealPrepState)

# Register our execution nodes
builder.add_node("load_data", load_data_files)
builder.add_node("fetch_context", fetch_environment_context)
builder.add_node("generate_plan", generate_meal_plan)

# Flow Chart Logic: Start -> Load Text Files -> Run Tools -> Ask LLM -> End
builder.set_entry_point("load_data")
builder.add_edge("load_data", "fetch_context")
builder.add_edge("fetch_context", "generate_plan")
builder.add_edge("generate_plan", END)

# Crucial: This variable 'workflow' must match what is written in your langgraph.json
workflow = builder.compile()
