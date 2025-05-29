import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

openai.api_key = OPENAI_API_KEY

# Load herbal knowledge JSON
with open("herbal_knowledge_synthesized.json", "r") as f:
    herbal_knowledge = json.load(f)

# (Optional) Pinecone setup (scaffold)
try:
    import pinecone
    if PINECONE_API_KEY and PINECONE_ENV:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        # Example: pinecone_index = pinecone.Index("herbal-index")
    else:
        pinecone = None
except ImportError:
    pinecone = None

app = FastAPI()

# Allow CORS for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Profile(BaseModel):
    id: str
    name: str
    allergies: list[str]
    chronicConditions: list[str]
    medications: list[str]

class QueryRequest(BaseModel):
    query: str
    profile: Profile

@app.post("/getHerbalistResponse")
async def get_herbalist_response(req: QueryRequest):
    # 1. Try to answer from herbal_knowledge (simple keyword search)
    query_lower = req.query.lower()
    found = None
    for key, value in herbal_knowledge.items():
        if query_lower in key.lower() or query_lower in str(value).lower():
            found = value
            break
    if found:
        return {
            "advice": found.get("advice", ""),
            "evidenceSnippets": found.get("evidenceSnippets", []),
            "preparationSteps": found.get("preparationSteps", ""),
            "safetyDosage": found.get("safetyDosage", ""),
            "freshnessDate": found.get("freshnessDate", ""),
            "followUpQuestions": found.get("followUpQuestions", [])
        }

    # 2. (Optional) Pinecone semantic search (scaffold)
    # if pinecone:
    #     ...

    # 3. Fall back to OpenAI
    prompt = (
        f"User profile: {req.profile.dict()}\n"
        f"User question: {req.query}\n"
        "As an empathetic, evidence-based herbalist, provide advice, evidence snippets, preparation steps, "
        "safety/dosage, and follow-up questions. Respond in JSON with keys: advice, evidenceSnippets, "
        "preparationSteps, safetyDosage, freshnessDate, followUpQuestions."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=500,
        temperature=0.7,
    )
    try:
        ai_json = json.loads(response.choices[0].message['content'])
    except Exception as e:
        return {"error": "AI response was not valid JSON", "details": str(e)}
    return ai_json