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

system_template = (
    "You are a discussion topic generator for forums. You will be given a topic keyword and tone of voice. Your task is to create a discussion topic that is simple, concise, and human-like. Keep the response to a maximum of 50 words. Use everyday words that a child could understand. Avoid slang, jargon, or complex words."
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "Please generate a topic using the keyword '{topic}' in a '{tone}' tone of voice.")
    ]
)

gen_topic_chain = prompt_template | model

class TopicRequest(BaseModel):
    id: str
    topic: str
    tone: str

class TopicResponse(BaseModel):
    id: str
    generated_topic: str

STATIC_TOKEN = os.getenv("STATIC_TOKEN", "default_static_token")

async def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    if authorization != f"Bearer {STATIC_TOKEN}":
        raise HTTPException(status_code=403, detail="Invalid or missing token")
    return True

@app.post("/generate-topic", response_model=TopicResponse, dependencies=[Depends(verify_token)])
async def generate_topic(request: TopicRequest):
    """
    Generate a discussion topic based on the given topic and tone.
    """
    try:
        result = gen_topic_chain.invoke({"topic": request.topic, "tone": request.tone})
        return TopicResponse(id=request.id, generated_topic=result.content.replace('"', ''))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app)
