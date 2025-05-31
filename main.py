from fastapi import FastAPI, UploadFile, File
from google import genai
from fastapi.responses import JSONResponse
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware


class StoryRequest(BaseModel):
    prompt: str
    Style: str

load_dotenv()

app = FastAPI()

client = genai.Client(api_key=os.getenv("GEMINI_API"))

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to transcribe audio files.
    """
    if not file.filename.endswith(".mp3"):
        return {"error": "Only .mp3 files are allowed."}
    filaudio_bytes = await file.read()
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            'transcribe this audio clip in English language',
            types.Part.from_bytes(
                data=filaudio_bytes,
                mime_type= "audio/mp3",
            )
        ]
        )
    print(response.text)
    return JSONResponse(content={ "filename": file.filename, "Content": file.content_type, "transcribe": response.text})


@app.post("/generate-story-gemini")
async def generate_story_gemini(storyPrompt: StoryRequest):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
                system_instruction=[
                    types.Part.from_text(text=f"You are a helpful assistant that generates stories based on prompts story style should be {storyPrompt.Style}."),
                ],
                temperature=0
            ),
        contents = [
            types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=storyPrompt.prompt),
                ],
            ),
        ]
    )
    # print(response.text)
    return JSONResponse(content={"story": response.text})


@app.post("/generate-story-langchain")
async def generate_story_langchain(storyPrompt: StoryRequest):
    """
    Endpoint to generate a story using LangChain.
    """
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_key = os.getenv("GEMINI_API"))
    messages = [
        ("system", f"You are a helpful assistant that generates stories based on prompts story style should be {storyPrompt.Style}."),
        ("human", f"{storyPrompt.prompt}."),
    ]
    return JSONResponse(content={ "llm": llm.invoke(messages).content })