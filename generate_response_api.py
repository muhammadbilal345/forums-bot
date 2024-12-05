import os
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ChatGroq(
    temperature=0.6,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile"
)

system_response_template = (
    "You are a response generator for forums. You will be given a query that you will analyze. Your task is to provide an answer or more suggestion like human do that should be simple, concise, and human-like. Keep the response to a maximum of 50 words. Use everyday words that a child could understand. Avoid slang, jargon, or complex words."
)

system_moderation_template = (
    "You are a content moderator. Analyze the given text and determine if it contains harmful, abusive, or inappropriate language. "
    "Respond with 'delete' if it contains such language; otherwise, respond with 'safe'."
)

response_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_response_template),
        ("user", "Please generate a response using the text which is '{text}' in a '{tone}' tone of voice.")
    ]
)

moderation_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_moderation_template),
        ("user", "Analyze this text: '{text}'")
    ]
)

response_chain = response_prompt_template | model
moderation_chain = moderation_prompt_template | model

class TextRequest(BaseModel):
    id: str
    text: str
    tone: str

class TextResponse(BaseModel):
    id: str
    generated_response: str

STATIC_TOKEN = os.getenv("STATIC_TOKEN", "default_static_token")

async def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    if authorization != f"Bearer {STATIC_TOKEN}":
        raise HTTPException(status_code=403, detail="Invalid or missing token")
    return True

@app.post("/generate-response", response_model=TextResponse, dependencies=[Depends(verify_token)])
async def generate_response(request: TextRequest):
    """
    Generate a response based on the given text and tone, with moderation.
    """
    try:
        moderation_result = moderation_chain.invoke({"text": request.text})
        moderation_decision = moderation_result.content.strip().lower()
        
        if moderation_decision == "delete":
            return TextResponse(id=request.id, generated_response="delete")
        
        result = response_chain.invoke({"text": request.text, "tone": request.tone})
        return TextResponse(id=request.id, generated_response=result.content.replace('"', ''))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app)
