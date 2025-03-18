import os
import google.generativeai as genai
from dotenv import load_dotenv
from main_rag import RAGSystem
from pinecone.grpc import PineconeGRPC

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05') 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if PINECONE_API_KEY:
    pc = PineconeGRPC(api_key=PINECONE_API_KEY)
else:
    pc = None
    print("Pinecone API key not found in environment variables.")

# Initialize RAG system
rag = RAGSystem(
    pc_client=pc,
    index_name="studio",
    gemini_api_key=GEMINI_API_KEY,
    dimension=1024,
    embedding_model_name="BAAI/bge-large-en-v1.5"
)

docx_file_path = r"v1chap2.docx"

# if os.path.exists(docx_file_path):
#     rag.index_document_by_sentences(docx_file_path, chunk_size= 3, chunk_overlap= 1, source="v1chap2-doc")

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
            "description": "Provide responses to use queries that are not for retrieving information from materials and textbooks. Use your general knowledge to provide explanations to information from the materials and textbooks retrieved. Be very brief and concise with your responses",
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

controller_tools = [
    {
        "function_declarations": [{
            "name": "clarification",
            "description": "This tool Informs the first AI on what needs to be done and adjusted in its responses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The first AI's response."
                    }
                },
                "required": ["query"]
            }
        }]
    },
    {
        "function_declarations": [{
            "name": "final_response",
            "description": "Provide a final and intelligent response to the user if the first AI response is okay.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The first AI's response."
                    }
                },
                "required": ["query"]
            }
        }]
    }
]
# ------------------------------------------------------------------------------
# Role Prompt for the Supervisor Agents
# ------------------------------------------------------------------------------
role_prompt = """
You are a Super Intelligent Chatbot with Advanced Capabilities built for students to help them access key information from their materials and textbooks.
You MUST ALWAYS use one of the provided tools to respond to the user. DO NOT provide any response directly without using a tool.
Your available tools are:
    - 'perform_rag': Use this for questions related to your learning materials and textbooks.
    - 'generic_response': Use this for any other type of question.

If the user's query is unclear, or you are unsure which tool to use, you MUST use the 'generic_response' tool to ask for clarification.  The user's query MUST be passed to the tool.

Under NO circumstances should you provide a response that does not originate from one of these tools.  Every response MUST be the result of a function call.
"""

controller_prompt = """
You are a  controller agent for STUDio and you are responsible for ensuring that the user gets the best of answers to their queries.\
You are to ensure that when the first AI gives a response, you are to check the response and ensure that the response is accurate and helpful to the user and the appropriate tool was called by the first AI.\
If the first AI didn't give a befitting response, you are to instruct the first AI to call the appropriate tool to give the user a better response.\
You are to ensure that the first AI is always in check and that the user gets the best of responses to their queries.\
These are the tools that the first AI can use to give the user the best of responses:
    - 'perform_rag': Use this for questions related to your learning materials and textbooks.
    - 'generic_response': Use this for any other type of question.

Use the 'clarification' tool to instruct the first AI to use the appropriate tool to give the user a better response.\
Use the 'final_response' tool to instruct the to give the user a final and intelligent response if the first AI response is okay.\
Also, add more information to the response of the first AI to make the response more helpful to the user.\
If the user's query is unclear, or you are unsure which tool to use, you MUST insturst the first AI to use the 'generic_response' tool to ask for clarification.  The user's query MUST be passed to the tool.
If the first AI's response to correct, instruct the first A
    """

# ------------------------------------------------------------------------------
# Function to Extract the Function Call from Gemini Model
# ------------------------------------------------------------------------------
def extract_function_call(user_query: str):
    messages = [
        {"role": "user", "parts": [user_query, role_prompt]}
    ]
    response = model.generate_content(messages, tools=tools)
    # Correctly access the function call information
    if response.candidates and response.candidates[0].content.parts:
        part = response.candidates[0].content.parts[0]
        if part.function_call:
            func_call = part.function_call
            func_name = func_call.name
            args = func_call.args
            print (args)
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
    conversation_context = "\n".join([f"User: {entry['user']}\n {entry['response']}" for entry in query_history])
    full_prompt = f"{conversation_context}\nUser: {query_text}" if conversation_context else query_text

    if func_name == "perform_rag":
        result = result = rag.query(full_prompt)
        query_history.append({"user": query_text, "response": result})
        return result

    elif func_name == "generic_response":
        response = model.generate_content(full_prompt)
        query_history.append({"user": query_text, "response": response.text})
        return response.text

    else:
        return response
    

# def get_final_response(query):
#     mid_response = handle_user_query(query)
#     is_satisfactory = False
#     iteration = 0

#     while not is_satisfactory and iteration < 5:  # Allow up to 5 iterations
#         iteration += 1

#         # Create the message that contains the query, the AI response, and the controller instructions
#         messages = [
#             {"role": "user", "parts": [query, mid_response, controller_prompt]}
#         ]

#         # Send the messages to the model to generate content
#         response = model.generate_content(messages, tools=controller_tools)

#         # Check if the response is useful
#         if response.candidates and response.candidates[0].content.parts:
#             part = response.candidates[0].content.parts[0]
#             print(part)
#             if part.function_call:
#                 func_call = part.function_call
#                 func_name = func_call.name
#                 args = func_call.args

#                 # If the response needs improvement, instruct the first AI
#                 if func_name == "clarification" :
#                     mid_response = handle_user_query(response)
#                     is_satisfactory = False  # Await clarification and retry
#                 elif func_name == "final_response":
#                     final_response = response.text
#                     is_satisfactory = True
#                 else:
#                     # If the response is satisfactory, finalize the answer
#                     final_response = response.text
#                     is_satisfactory = True
#             else:
#                 # No function call, directly take the response as it is
#                 final_response = response.text
#                 is_satisfactory = True
        
#         # If iteration limit reached, return a fallback response
#         if iteration == 5:
#             final_response = "Unable to generate a satisfactory response after several attempts."

#     return final_response

# ------------------------------------------------------------------------------
# Testing the Supervisor Agent
# ------------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Test RAG Query (Make sure main_rag.py and your RAG data are set up!)
#     response = handle_user_query("How do I resolve company already onboarded, 409 error?")
#     print(response)

    # # Test Generic Query
    # response = handle_user_query("What is the capital of France?", "Jane Smith")
    # print(response)