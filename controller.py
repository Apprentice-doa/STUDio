import os
import google.generativeai as genai
from dotenv import load_dotenv
from main_rag import rag  

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro') 

# ------------------------------------------------------------------------------
# Define Tool Specifications for Function Calling
# ------------------------------------------------------------------------------
tools = [
     {
        "function_declarations": [{
            "name": "perform_rag",
            "description": "Retrieve information from materials and textbooks to help students learn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User query asking for information from a material."
                    }
                },
                "required": ["query"]
            }
        }]
    },
    {
        "function_declarations": [{
            "name": "generic_response",
            "description": "Provide a generic response for general queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "General user query."
                    }
                },
                "required": ["query"]
            }
        }]
    }
]

# ------------------------------------------------------------------------------
# Role Prompt for the Supervisor Agent
# ------------------------------------------------------------------------------
role_prompt = """
You are a Super Intelligent Chatbot with Advanced Capabilities built for students to help them access key information from their materials and textbooks. 
You help users by retrieving information or setting timers for reading based on their requests. 
- If the student asks about general information, use the RAG agent.
- If the question is generic, use the generic response tool.
- If unsure, ask the student for clarification.
"""

# ------------------------------------------------------------------------------
# Function to Extract the Function Call from OpenAI
# ------------------------------------------------------------------------------
def extract_function_call(user_query: str):
    messages = [
        {"role": "user", "parts": [user_query]},
        {"role": "model", "parts": ["Okay, I'll help you with that."]}, # Add a model part for the conversation to work
    ]

    response = model.generate_content(messages, tools=tools)

    # Correctly access the function call information
    if response.candidates and response.candidates[0].content.parts:
        part = response.candidates[0].content.parts[0]
        if part.function_call:
            func_call = part.function_call
            func_name = func_call.name
            args = func_call.args
            return func_name, args

    return None, None


# ------------------------------------------------------------------------------
# Function to Handle User Queries
# ------------------------------------------------------------------------------
query_history = []
def handle_user_query(prompt):
    func_name, args = extract_function_call(prompt)
    print (func_name)
    print (args)
    if args is None:
        args = {}  # Ensure args is a dictionary

    print(f"Detected function call: {func_name} with args: {args}") 
    response = ""
    query_text = args.get("query", "")
    conversation_context = "\n".join([f"User: {entry['user']}\nAssistant: {entry['response']}" for entry in query_history])
    full_prompt = f"{conversation_context}\nUser: {query_text}" if conversation_context else query_text

    if func_name == "perform_rag":
        result = result = rag.query(full_prompt)
        query_history.append({"user": query_text, "response": response})
        return result

    elif func_name == "generic_response":
        response = model.generate_content(full_prompt)
        query_history.append({"user": query_text, "response": response.text})
        return response.text

    else:
        return response

# ------------------------------------------------------------------------------
# Testing the Supervisor Agent
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test RAG Query (Make sure main_rag.py and your RAG data are set up!)
    response = handle_user_query("How do I resolve company already onboarded, 409 error?")
    print(response)

    # # Test Generic Query
    # response = handle_user_query("What is the capital of France?", "Jane Smith")
    # print(response)