import os
import json
import time
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator
from openai import OpenAI
from typing import Optional, Dict, Tuple
import firebase_admin
from firebase_admin import credentials, auth

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debug logging for API key
if OPENAI_API_KEY:
    print(f"API key loaded: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-4:]}")
else:
    print("WARNING: OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Firebase Admin SDK
firebase_configured = False
try:
    # Try to get Firebase credentials from environment
    firebase_creds = os.getenv("FIREBASE_CREDENTIALS")
    print(f"[DEBUG] FIREBASE_CREDENTIALS environment variable: {'Found' if firebase_creds else 'Not found'}")
    
    if firebase_creds:
        print(f"[DEBUG] Firebase credentials length: {len(firebase_creds)} characters")
        print(f"[DEBUG] Firebase credentials preview: {firebase_creds[:50]}...")
        
        try:
            cred_dict = json.loads(firebase_creds)
            print(f"[DEBUG] Firebase credentials parsed successfully, project_id: {cred_dict.get('project_id', 'N/A')}")
            
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            firebase_configured = True
            print("Firebase Admin SDK initialized successfully")
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Firebase credentials JSON parsing error: {e}")
            firebase_configured = False
        except Exception as e:
            print(f"[DEBUG] Firebase certificate creation error: {e}")
            firebase_configured = False
    else:
        print("Warning: FIREBASE_CREDENTIALS environment variable not found")
        firebase_configured = False
except Exception as e:
    print(f"Warning: Firebase initialization failed: {e}")
    firebase_configured = False

print(f"[DEBUG] Final Firebase configuration status: {firebase_configured}")

# Simple in-memory rate limiting (for serverless compatibility)
rate_limit_store: Dict[str, Tuple[int, float]] = {}

def check_rate_limit(client_id: str, limit: int, window_seconds: int) -> bool:
    """Simple rate limiting using in-memory storage"""
    current_time = time.time()
    
    if client_id in rate_limit_store:
        count, window_start = rate_limit_store[client_id]
        
        # Check if we're still in the same window
        if current_time - window_start < window_seconds:
            # Still in window, check count
            if count >= limit:
                return False
            # Increment count
            rate_limit_store[client_id] = (count + 1, window_start)
        else:
            # New window, reset
            rate_limit_store[client_id] = (1, current_time)
    else:
        # First request
        rate_limit_store[client_id] = (1, current_time)
    
    return True

# Load herbal knowledge JSON
try:
    with open("herbal_knowledge_synthesized.json", "r") as f:
        herbal_knowledge = json.load(f)
except FileNotFoundError:
    print("Warning: herbal_knowledge_synthesized.json not found")
    herbal_knowledge = {}
except json.JSONDecodeError:
    print("Warning: Invalid JSON in herbal_knowledge_synthesized.json")
    herbal_knowledge = {}
except Exception as e:
    print(f"Warning: Error loading herbal knowledge: {e}")
    herbal_knowledge = {}

# Search functions for herbal knowledge
def extract_keywords(query):
    """Extract relevant keywords from user query"""
    query_lower = query.lower()
    
    # Define symptom keywords mapping
    symptom_keywords = {
        "cold": ["cold", "flu", "immune", "respiratory", "congestion", "fever", "sore throat"],
        "headache": ["headache", "migraine", "pain", "tension", "head pain"],
        "anxiety": ["anxiety", "stress", "nervous", "calming", "relaxation", "worry"],
        "digestive": ["digestive", "stomach", "nausea", "indigestion", "bloating", "constipation"],
        "sleep": ["sleep", "insomnia", "rest", "tired", "fatigue"],
        "immune": ["immune", "infection", "virus", "bacteria", "sick"],
        "pain": ["pain", "ache", "sore", "inflammation", "swelling"],
        "energy": ["energy", "tired", "fatigue", "vitality", "stamina"],
        "skin": ["skin", "acne", "rash", "eczema", "clear skin"],
        "heart": ["heart", "cardiovascular", "blood pressure", "circulation"],
        "liver": ["liver", "detox", "cleanse", "toxins"],
        "kidney": ["kidney", "urinary", "bladder", "diuretic"],
        "bone": ["bone", "joint", "arthritis", "calcium", "density"],
        "vision": ["vision", "eye", "sight", "eyesight", "clear vision"]
    }
    
    # Find matching symptom categories
    keywords = []
    for symptom, related_words in symptom_keywords.items():
        if any(word in query_lower for word in related_words):
            keywords.extend(related_words)
    
    # Add original query words
    keywords.extend(query_lower.split())
    
    return list(set(keywords))  # Remove duplicates

def search_herbal_knowledge(query, herbal_data):
    """Search herbal knowledge for relevant herbs"""
    keywords = extract_keywords(query)
    relevant_herbs = []
    
    for herb_name, herb_data in herbal_data.items():
        # Skip category entries (they don't have individual herb properties)
        if herb_name in ["Herbal Medicine", "Immune System Support", "Digestive Health", 
                        "Mental and Emotional Health", "Physical Health and Vitality",
                        "Introduction to Herbalism", "Herbalism in Traditional Chinese Medicine (TCM)",
                        "Herbalism in Ayurveda", "Herbalism in Greco-Roman Tradition",
                        "Herbalism in Indigenous Peoples of the Americas", "error"]:
            continue
            
        # Calculate relevance score
        score = 0
        herb_text = f"{herb_name} {herb_data.get('Properties', '')} {herb_data.get('Uses', '')}".lower()
        
        for keyword in keywords:
            if keyword in herb_text:
                score += 1
                
        # Bonus points for exact matches
        if any(keyword in herb_name.lower() for keyword in keywords):
            score += 2
            
        if score > 0:
            relevant_herbs.append({
                "name": herb_name,
                "data": herb_data,
                "score": score
            })
    
    # Sort by relevance score and return top 5
    relevant_herbs.sort(key=lambda x: x["score"], reverse=True)
    return relevant_herbs[:5]

# Conversation memory storage (in production, this would be in a database)
conversation_memory = {}

async def detect_conversation_phase(user_query, user_id, conversation_context):
    """Detect which phase of conversation we're in using AI-powered analysis"""
    
    # Extract conversation context from iOS app
    diagnostic_count = 0
    last_phase = "initial"
    
    if conversation_context:
        diagnostic_count = conversation_context.get("diagnosticQuestionCount", 0)
        last_phase = conversation_context.get("lastPhase", "initial")
    
    print(f"[DEBUG] AI Phase Detection - Count: {diagnostic_count}, Last: {last_phase}, Query: {user_query[:50]}...")
    
    # Use AI to analyze the message and determine phase
    try:
        # Call the enhanced analyzeIntent endpoint internally
        ai_analysis = await analyze_intent_internal({
            "message": user_query,
            "context": str(conversation_context)
        }, user_id)
        
        # Extract phase from AI analysis
        phase = ai_analysis.get("phase", "diagnostic")
        ready_for_remedy = ai_analysis.get("ready_for_remedy", False)
        symptoms = ai_analysis.get("symptoms", [])
        
        print(f"[DEBUG] AI Phase Analysis - Phase: '{phase}', Ready: {ready_for_remedy}, Symptoms: {symptoms}")
        
        # Override phase based on AI analysis
        if phase == "recommendation" and ready_for_remedy:
            print("[DEBUG] AI determined ready for recommendation")
            return "recommendation"
        elif phase == "recommendation":
            print("[DEBUG] AI determined recommendation phase (even without ready_for_remedy)")
            return "recommendation"
        elif phase == "follow_up":
            print("[DEBUG] AI detected follow_up phase")
            return "follow_up"
        elif phase == "general":
            print("[DEBUG] AI detected general phase")
            return "general_knowledge"
        else:
            print("[DEBUG] AI determined diagnostic phase needed")
            return "diagnostic"
            
    except Exception as e:
        print(f"[DEBUG] AI phase detection failed: {e}, falling back to rule-based")
        # Fallback to simple rule-based detection
        if "side effects" in user_query.lower() or "dosage" in user_query.lower():
            return "follow_up"
        elif "what is" in user_query.lower() or "tell me about" in user_query.lower():
            return "general_knowledge"
        else:
            return "diagnostic"

async def analyze_intent_internal(request_data, user_id):
    """Internal function to analyze intent without HTTP overhead"""
    try:
        user_message = request_data.get("message", "")
        context = request_data.get("context", "")
        
        # AI prompt for intent and phase detection
        ai_prompt = f"""
You are an AI assistant that analyzes user messages for a herbal medicine app. Your job is to classify the user's intent and conversation phase.

User message: "{user_message}"
Context: "{context}"

INTENT CLASSIFICATION RULES:
1. "remedy" = ANY mention of symptoms, health issues, or requests for help
2. "alternative" = asking for different treatment options
3. "track" = wanting to track progress
4. "reminder" = wanting reminders
5. "general" = ONLY questions about herbs/wellness knowledge (NOT symptoms)

PHASE CLASSIFICATION RULES:
1. "diagnostic" = first mention of symptoms, need more info
2. "recommendation" = ready to provide remedies (sufficient symptom detail)
3. "follow_up" = questions about specific herbs/treatments
4. "general" = general knowledge questions

EXAMPLES:
- "I have a headache" → intent: "remedy", phase: "diagnostic"
- "I woke up with congestion" → intent: "remedy", phase: "diagnostic"  
- "I have had severe sinus pressure for 3 hours" → intent: "remedy", phase: "recommendation"
- "What is peppermint good for?" → intent: "general", phase: "general"
- "I need help with my stomach ache" → intent: "remedy", phase: "diagnostic"

Respond with ONLY a JSON object in this exact format:
{{
    "intent": "remedy|alternative|track|reminder|general",
    "phase": "diagnostic|recommendation|follow_up|general", 
    "symptoms": ["symptom1", "symptom2"],
    "ready_for_remedy": true|false
}}

REMEMBER: ANY symptom mention = "remedy" intent. ONLY herb knowledge questions = "general" intent.
"""
        
        # Use OpenAI to analyze
        ai_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": ai_prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        
        # Parse JSON response
        response_text = ai_response.choices[0].message.content.strip()
        result = json.loads(response_text)
        
        # Validate and extract fields
        intent = result.get("intent", "general")
        phase = result.get("phase", "diagnostic")
        symptoms = result.get("symptoms", [])
        ready_for_remedy = result.get("ready_for_remedy", False)
        
        # Validate intent
        valid_intents = ["remedy", "alternative", "track", "reminder", "general"]
        if intent not in valid_intents:
            intent = "general"
        
        # Validate phase
        valid_phases = ["diagnostic", "recommendation", "follow_up", "general"]
        if phase not in valid_phases:
            phase = "diagnostic"
        
        return {
            "intent": intent,
            "phase": phase,
            "symptoms": symptoms,
            "ready_for_remedy": ready_for_remedy,
            "confidence": "high"
        }
        
    except Exception as e:
        print(f"[DEBUG] Internal AI analysis error: {e}")
        return {
            "intent": "general",
            "phase": "diagnostic",
            "symptoms": [],
            "ready_for_remedy": False,
            "confidence": "low"
        }

def determine_knowledge_source(user_query, phase):
    """Determine whether to use curated knowledge or general AI knowledge"""
    if phase == "recommendation":
        return "curated_herbal_knowledge"
    elif phase == "follow_up":
        return "curated_herbal_knowledge"
    elif "remedy" in user_query or "recommend" in user_query:
        return "curated_herbal_knowledge"
    elif "what is" in user_query or "tell me about" in user_query:
        return "general_ai_knowledge"
    elif "how to use" in user_query or "dosage" in user_query:
        return "curated_herbal_knowledge"
    else:
        return "general_ai_knowledge"

def create_personalized_prompt(user_query, user_profile, phase, user_id, conversation_context):
    """Create a personalized prompt based on conversation phase and memory"""
    user_memory = conversation_memory.get(user_id, {})
    
    # Build memory context
    memory_context = ""
    if user_memory.get("diagnostic_history"):
        memory_context += f"\nPrevious consultations: {user_memory['diagnostic_history']}\n"
    if user_memory.get("preferences"):
        memory_context += f"\nUser preferences: {user_memory['preferences']}\n"
    if user_memory.get("recent_conversations"):
        memory_context += f"\nRecent topics: {', '.join(user_memory['recent_conversations'][-3:])}\n"
    
    if phase == "diagnostic":
        return f"""
You are Herba, a warm, expert AI herbalist. The user is in the diagnostic phase.

User's response: {user_query}
User profile: {user_profile}
{memory_context}

Based on their answer, ask ONE follow-up diagnostic question to gather more information about their symptoms. 
Focus on severity, duration, or specific symptoms that would help recommend the best herbal remedy.

IMPORTANT: 
- Respond ONLY with the diagnostic question
- Do not provide recommendations yet
- Keep the question conversational and natural
- Ask about severity, duration, or specific symptoms
- Be empathetic and warm in tone
- Reference previous consultations if relevant

Example responses:
- "How severe would you rate your symptoms on a scale of 1-10?"
- "How long have you been experiencing these symptoms?"
- "Have you experienced this type of pain before?"
- "Are you currently taking any medications?"
"""

    elif phase == "recommendation":
        # Use diagnostic context for personalized recommendations
        diagnostic_context = user_memory.get("current_diagnostic", {})
        relevant_herbs = search_herbal_knowledge(user_query, herbal_knowledge)
        
        return f"""
You are Herba, an expert AI herbalist. Provide personalized herbal recommendations.

User query: {user_query}
User profile: {user_profile}
Diagnostic context: {diagnostic_context}
{memory_context}

Relevant herbs: {relevant_herbs}

CRITICAL INSTRUCTIONS:
- You MUST respond with ONLY valid JSON format
- NO natural language text before or after the JSON
- NO tracking questions or progress monitoring offers
- NO "Would you like to track" or similar phrases
- Focus ONLY on the remedy recommendation and instructions
- If you cannot provide JSON, respond with: {{"error": "Unable to provide recommendation in required format"}}

Use this EXACT JSON format (no variations):

{{
  "recommendations": [
    {{
      "herb": "Single Best Herb Name",
      "benefits": "How this herb specifically helps with their condition",
      "dosage": "Specific dosage and preparation instructions",
      "safety_notes": "Important safety warnings and precautions"
    }}
  ],
  "safety_considerations": "Overall safety considerations for the user",
  "interactions": "Potential medication interactions to be aware of",
  "timeline": "Expected timeline for seeing benefits",
  "additional_advice": "Lifestyle recommendations and additional tips"
}}
"""

    elif phase == "follow_up":
        current_recommendation = user_memory.get("current_recommendation", {})
        return f"""
You are Herba, an expert AI herbalist. Answer specific questions about the recommended remedy.

User question: {user_query}
User profile: {user_profile}
Current recommendation: {current_recommendation}
{memory_context}

Provide detailed, personalized information about the herb, considering the user's specific situation.
Be warm, informative, and reference their diagnostic context when relevant.
"""

    else:  # general_knowledge
        return f"""
You are Herba, a warm, expert AI herbalist. Answer general herbal knowledge questions.

User question: {user_query}
User profile: {user_profile}
{memory_context}

Provide educational information about herbs and natural wellness.
Be conversational, informative, and reference previous conversations when relevant.
Keep responses natural and engaging.
"""

def update_conversation_memory(user_id, user_query, ai_response, phase, conversation_context):
    """Update conversation memory with new information"""
    if user_id not in conversation_memory:
        conversation_memory[user_id] = {
            "diagnostic_history": {},
            "preferences": {},
            "recent_conversations": [],
            "current_diagnostic": {},
            "current_recommendation": {}
        }
    
    user_memory = conversation_memory[user_id]
    
    # Update recent conversations
    user_memory["recent_conversations"].append(f"{user_query[:50]}...")
    if len(user_memory["recent_conversations"]) > 10:
        user_memory["recent_conversations"] = user_memory["recent_conversations"][-10:]
    
    # Update diagnostic context
    if phase == "diagnostic":
        # Extract diagnostic information from user query
        if "severity" in ai_response.lower():
            user_memory["current_diagnostic"]["severity"] = user_query
        elif "duration" in ai_response.lower():
            user_memory["current_diagnostic"]["duration"] = user_query
        elif "experience" in ai_response.lower():
            user_memory["current_diagnostic"]["experience"] = user_query
    
    # Mark diagnostic as complete after 2-3 questions
    if phase == "diagnostic" and len(user_memory["current_diagnostic"]) >= 2:
        user_memory["diagnostic_complete"] = True
    
    # Store recommendation when given
    if phase == "recommendation":
        user_memory["current_recommendation"] = {
            "query": user_query,
            "response": ai_response,
            "timestamp": "2024-08-07"
        }

app = FastAPI(title="Herba AI Proxy", version="1.0.0")

# Improved CORS - restrict to your iOS app's origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://herba-ai-proxy.vercel.app",  # Your Vercel domain
        "http://localhost:3000",  # For local development
        "http://localhost:8000",  # For local development
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Input validation models
class Profile(BaseModel):
    id: str
    name: str
    allergies: list[str]
    chronicConditions: list[str]
    medications: list[str]
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Name cannot be empty')
        if len(v) > 100:
            raise ValueError('Name too long (max 100 characters)')
        return v.strip()
    
    @validator('allergies', 'chronicConditions', 'medications')
    def validate_lists(cls, v):
        if len(v) > 50:  # Prevent excessive list sizes
            raise ValueError('List too long (max 50 items)')
        return v

class ConversationContext(BaseModel):
    diagnosticQuestionCount: int = 0
    lastPhase: str = "initial"
    diagnosticContext: dict = {}
    messageCount: int = 0

class QueryRequest(BaseModel):
    query: str
    profile: Profile
    conversationContext: Optional[ConversationContext] = None
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:  # Prevent extremely long queries
            raise ValueError('Query too long (max 1000 characters)')
        return v.strip()

async def verify_firebase_token(authorization: Optional[str] = Header(None)):
    """Verify Firebase ID token and extract user ID"""
    print(f"[DEBUG] Firebase configured: {firebase_configured}")
    print(f"[DEBUG] Authorization header present: {authorization is not None}")
    
    if not firebase_configured:
        # If Firebase is not configured, return a default user ID for testing
        print("[DEBUG] Firebase not configured, using default user ID")
        return "default_user"
    
    if not authorization:
        print("[DEBUG] No authorization header provided")
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        # Remove 'Bearer ' prefix if present
        token = authorization.replace('Bearer ', '') if authorization.startswith('Bearer ') else authorization
        print(f"[DEBUG] Token length: {len(token)}")
        print(f"[DEBUG] Token preview: {token[:20]}...{token[-10:] if len(token) > 30 else token}")
        
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        
        print(f"[DEBUG] Successfully verified token for user: {user_id}")
        print(f"[DEBUG] User email: {decoded_token.get('email', 'N/A')}")
        
        return user_id
    except auth.InvalidIdTokenError as e:
        print(f"[DEBUG] Invalid Firebase token: {e}")
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    except auth.ExpiredIdTokenError as e:
        print(f"[DEBUG] Expired Firebase token: {e}")
        raise HTTPException(status_code=401, detail="Firebase token expired")
    except Exception as e:
        print(f"[DEBUG] Firebase token verification error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=401, detail=f"Firebase token verification failed: {str(e)}")

@app.post("/getHerbalistResponse")
async def get_herbalist_response(
    req: QueryRequest,
    user_id: str = Depends(verify_firebase_token)
):
    print(f"[DEBUG] 🆔 User ID extracted: {user_id}")
    print(f"[DEBUG] 📝 Query: {req.query[:100]}...")
    print(f"[DEBUG] 👤 Profile name: {req.profile.name}")
    
    # Rate limiting: 10 requests per minute per user
    if not check_rate_limit(user_id, 10, 60):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please wait before making another request."
        )
    
    try:
        # Validate input
        if not req.query or len(req.query.strip()) == 0:
            raise HTTPException(status_code=422, detail="Query cannot be empty")
        
        if not req.profile or not req.profile.name:
            raise HTTPException(status_code=422, detail="Profile name is required")
        
        # Create user profile string
        profile_info = f"Name: {req.profile.name}"
        if req.profile.allergies:
            profile_info += f", Allergies: {', '.join(req.profile.allergies)}"
        if req.profile.chronicConditions:
            profile_info += f", Chronic Conditions: {', '.join(req.profile.chronicConditions)}"
        if req.profile.medications:
            profile_info += f", Medications: {', '.join(req.profile.medications)}"
        
        # Extract conversation context from iOS app
        conversation_context = {}
        print(f"[DEBUG] Raw conversationContext from request: {req.conversationContext}")
        
        if req.conversationContext:
            conversation_context = {
                "diagnosticQuestionCount": req.conversationContext.diagnosticQuestionCount,
                "lastPhase": req.conversationContext.lastPhase,
                "diagnosticContext": req.conversationContext.diagnosticContext,
                "messageCount": req.conversationContext.messageCount
            }
            print(f"[DEBUG] ✅ Conversation context extracted successfully: {conversation_context}")
        else:
            print("[DEBUG] ❌ No conversation context provided in request")
            conversation_context = {
                "diagnosticQuestionCount": 0,
                "lastPhase": "initial",
                "diagnosticContext": {},
                "messageCount": 0
            }
            print(f"[DEBUG] ℹ️ Using default conversation context: {conversation_context}")
        
        # Detect conversation phase using AI-powered analysis
        phase = await detect_conversation_phase(req.query, user_id, conversation_context)
        
        # Create personalized prompt
        prompt = create_personalized_prompt(req.query, profile_info, phase, user_id, conversation_context)
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Faster and more efficient than gpt-4
            messages=[{"role": "system", "content": prompt}],
            max_tokens=400,  # Optimized for faster responses
            temperature=0.7,
            stream=True,  # Enable streaming for real-time responses
        )
        
        # Stream the response
        def generate_stream():
            full_response = ""
            try:
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        # Send each chunk as Server-Sent Event
                        yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"
                
                # Update conversation memory with complete response
                update_conversation_memory(user_id, req.query, full_response, phase, conversation_context)
                
                # Check if response contains a remedy recommendation
                is_solution = phase == "recommendation" or "remedy" in full_response.lower() or any(herb in full_response.lower() for herb in ["echinacea", "peppermint", "ginger", "chamomile", "lavender", "turmeric", "aloe", "garlic", "honey", "lemon"])
                
                # Send completion signal with phase info and solution flag
                yield f"data: {json.dumps({'content': '', 'done': True, 'phase': phase, 'full_response': full_response, 'isSolution': is_solution})}\n\n"
                
            except Exception as e:
                print(f"Error in streaming: {e}")
                yield f"data: {json.dumps({'error': 'Stream interrupted', 'done': True})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2024-08-04T18:30:00Z"}

@app.get("/test")
async def test_endpoint():
    return {"message": "API is working!", "firebase_configured": firebase_configured}

@app.get("/test-auth")
async def test_auth_endpoint(user_id: str = Depends(verify_firebase_token)):
    """Test endpoint to verify Firebase authentication is working"""
    return {
        "message": "Authentication successful!",
        "user_id": user_id,
        "firebase_configured": firebase_configured,
        "timestamp": "2024-08-07T23:30:00Z"
    }

@app.post("/analyzeIntent")
async def analyze_intent(
    req: dict,
    user_id: str = Depends(verify_firebase_token)
):
    """AI-powered analysis to detect user intent from their message"""
    try:
        # Rate limiting
        if not check_rate_limit(user_id, 20, 60):  # 20 requests per minute
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        user_message = req.get("message", "")
        context = req.get("context", "")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message text is required")
        
        # AI prompt for intent and phase detection
        ai_prompt = f"""
Analyze this user message comprehensively to determine both intent and conversation phase.

User message: "{user_message}"
Context: "{context}"

INTENT CATEGORIES:
- "remedy": User is explicitly asking for a herbal remedy recommendation (e.g., "I need a remedy for headache", "What herb should I take?", "Recommend something for my pain")
- "alternative": User is asking for an alternative remedy (e.g., "What else can I try?", "Alternative to peppermint", "Different remedy")
- "track": User wants to track their progress (e.g., "Track my progress", "Monitor my symptoms", "Keep track")
- "reminder": User wants reminders (e.g., "Remind me", "Daily reminder", "Set up reminders")
- "general": General questions, clarifications, or other conversation

CONVERSATION PHASES:
- "diagnostic": User is describing symptoms but we need more information (e.g., "I have a headache", "I feel congested", "My stomach hurts")
- "recommendation": Ready to provide herbal remedies (user has described symptoms with sufficient detail)
- "follow_up": User is asking about specific herbs/treatments (e.g., "How do I use this?", "What are the side effects?")
- "general": General questions about herbs/wellness

SYMPTOM ANALYSIS:
- Extract any symptoms mentioned (e.g., "congestion", "headache", "sinus pressure", "runny nose")
- Determine if symptoms are described with sufficient detail for a recommendation

Respond with ONLY a JSON object in this exact format:
{{
    "intent": "remedy|alternative|track|reminder|general",
    "phase": "diagnostic|recommendation|follow_up|general",
    "symptoms": ["symptom1", "symptom2"],
    "ready_for_remedy": true|false
}}

Guidelines:
- If user describes symptoms with duration/severity, set ready_for_remedy=true
- If user just mentions symptoms without context, set ready_for_remedy=false
- Be precise about intent vs phase distinction
- Consider natural language variations for symptoms
"""
        
        # Use OpenAI to analyze the intent and phase
        ai_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": ai_prompt}],
            max_tokens=200,
            temperature=0.1,  # Low temperature for consistent analysis
        )
        
        # Extract the result
        response_text = ai_response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            import json
            result = json.loads(response_text)
            
            # Validate and extract fields
            intent = result.get("intent", "general")
            phase = result.get("phase", "diagnostic")
            symptoms = result.get("symptoms", [])
            ready_for_remedy = result.get("ready_for_remedy", False)
            
            # Validate intent
            valid_intents = ["remedy", "alternative", "track", "reminder", "general"]
            if intent not in valid_intents:
                intent = "general"
            
            # Validate phase
            valid_phases = ["diagnostic", "recommendation", "follow_up", "general"]
            if phase not in valid_phases:
                phase = "diagnostic"
            
            print(f"[DEBUG] AI Analysis - Message: '{user_message}', Intent: '{intent}', Phase: '{phase}', Symptoms: {symptoms}, Ready: {ready_for_remedy}")
            
            return {
                "intent": intent,
                "phase": phase,
                "symptoms": symptoms,
                "ready_for_remedy": ready_for_remedy,
                "confidence": "high",
                "message_preview": user_message[:100] + "..." if len(user_message) > 100 else user_message
            }
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing error: {e}, Response: {response_text}")
            # Fallback to basic intent detection
            return {
                "intent": "general",
                "phase": "diagnostic",
                "symptoms": [],
                "ready_for_remedy": False,
                "confidence": "low",
                "message_preview": user_message[:100] + "..." if len(user_message) > 100 else user_message
            }
        
    except Exception as e:
        print(f"[DEBUG] AI Intent Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=f"Intent analysis failed: {str(e)}")

@app.post("/analyzeResponseForRemedy")
async def analyze_response_for_remedy(
    req: dict,
    user_id: str = Depends(verify_firebase_token)
):
    """AI-powered analysis to detect if a response contains a herbal remedy recommendation"""
    try:
        # Rate limiting
        if not check_rate_limit(user_id, 20, 60):  # 20 requests per minute
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        response_text = req.get("response", "")
        if not response_text:
            raise HTTPException(status_code=400, detail="Response text is required")
        
        # AI prompt for remedy detection
        ai_prompt = f"""
Analyze this AI response and determine if it contains a herbal remedy recommendation.

Response to analyze:
{response_text}

Instructions:
- Look for herbal remedy suggestions, recommendations, or treatments
- Check for herb names, usage instructions, or remedy descriptions
- Consider context about natural solutions or herbal treatments

Respond with ONLY "true" if the response contains a herbal remedy recommendation, or "false" if it does not.
"""
        
        # Use OpenAI to analyze the response
        ai_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": ai_prompt}],
            max_tokens=10,
            temperature=0.1,  # Low temperature for consistent analysis
        )
        
        # Extract the result
        result = ai_response.choices[0].message.content.strip().lower()
        contains_remedy = result == "true"
        
        print(f"[DEBUG] AI Analysis - Response length: {len(response_text)}, Contains remedy: {contains_remedy}")
        
        return {
            "contains_remedy": contains_remedy,
            "analysis_confidence": "high",
            "response_preview": response_text[:100] + "..." if len(response_text) > 100 else response_text
        }
        
    except Exception as e:
        print(f"[DEBUG] AI Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/rate-limit-info")
async def rate_limit_info():
    return {
        "rate_limits": {
            "ai_chat": "10 requests per minute per user",
            "ai_analysis": "20 requests per minute per user",
        },
        "authentication": "Firebase ID token required"
    } # Force redeploy - Mon Aug 11 16:55:31 EDT 2025
print("Force redeploy - deployment test")
