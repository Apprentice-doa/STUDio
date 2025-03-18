from fastapi import FastAPI
from controller import handle_user_query
from model import userQuery

app = FastAPI()

@app.post("/getUserQuery")
def getUserQuery(userRequest: userQuery):
    return handle_user_query(userRequest.root)


# Run- uvicorn fast_api:app --reload to spin up the FastAPI server for the application.