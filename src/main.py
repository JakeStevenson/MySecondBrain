# API stuff
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Python Niceities 
from pydantic import BaseModel

# My other code
from brain import learn, ask, read_web

# Models for my API
class Question(BaseModel):
    question: str

class Inform(BaseModel):
    fact: str

class Article(BaseModel):
    url: str

class Output(BaseModel):
    output: str

# Set up the app
app=FastAPI()

@app.get("/")
async def home():
    return "Server Running"

@app.post("/read_article")
async def read_article(input: Article):
    await read_web(input.url)
    return "Read " + input.url

@app.post("/learn_fact")
async def learn_fact(input: Inform):
    learn(input.fact)
    return "Learned '" + input.fact + "'"

@app.post("/ask_question")
async def ask_question(input: Question):
    answer = ask(input.question)
    json_answer = jsonable_encoder(answer)
    return JSONResponse(content=json_answer)


origins = [
    "<http://localhost>",
    "<http://localhost:5173>",
    "<http://127.0.0.0:5173>",
    "<http://0.0.0.0:5173>"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5173)