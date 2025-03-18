from pydantic import RootModel, BaseModel
from typing import Optional

class userQuery(RootModel):
    root: str

class userDetails(BaseModel):
    userName: str
    userPassword: str
    userEmail: Optional[str] = None
    
