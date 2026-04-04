import os
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv('dental')
SHEETS_URL = os.getenv('SHEETS_URL')
if not api_key:
    raise RuntimeError("Missing 'dental' API key in .env")
client = Groq(api_key=api_key)

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI()
app.state.limiter = limiter

# ── Rate limit error handler ──────────────────────────────────────────────────
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"reply": "Too many messages! Please wait a moment."}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Security Headers ──────────────────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# ── Serve static files ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/images/logo.png")
async def get_logo():
    return FileResponse("static/images/logo.png")

# ── Clinic Knowledge ──────────────────────────────────────────────────────────
CLINIC_KNOWLEDGE = """
DENTAL CLINIC - TAJ DENTALS 

=== ABOUT US ===
Taj Dentals is a modern dental clinic established in 2025.
We provide comprehensive dental care for the whole family.

=== SERVICES ===
1. General Dentistry: Checkups, Fillings, Root Canal (RCT), Extractions, X-rays
2. Cosmetic Dentistry: Teeth Whitening, Veneers, Smile Makeovers, Bonding
3. Orthodontics: Metal Braces, Clear Aligners, Retainers
4. Implants & Restorations: Dental Implants, Crowns, Bridges, Dentures
5. Pediatric Dentistry: Children checkups from age 2, Sealants, Fluoride
6. Emergency Care: Toothache, Broken tooth, Lost filling, Abscess

=== LOCATION ===
Address: Rawalpindi Road, Dadyal, AJK
Landmark: Near THQ Dadyal. Free parking available.

=== CEO ===
CEO of Taj Dentals is Mirza Muhammad Mehboob
who is also CEO of Mughal Construction Group
show ceo name some one specifically ask for ceo name

=== CONTACT ===
Phone: +92-320-5018079
WhatsApp: +92-320-5018079
Email: tajdentals01@gmail.com

=== WORKING HOURS ===
Clinic is open every day: 9:00 AM - 7:00 PM
Surgeon is unavailable on Fridays. But clinic is open on Friday.
Public Holidays: Closed (Emergency contact available)

=== DOCTORS ===
Dr. Iqra Mehboob - Dental Surgeon, BDS RDS, 5 years experience
Mr. Qamar - Dental Hygienist and Dental Technician, 7 years experience
Mr. Ali - Dental Assistant, 3 years experience



=== SPECIAL OFFERS ===
Free first consultation for new patients
Family package: 10% discount
Senior citizens: 15% discount
Student discount: 10% off with valid ID

=== FAQs ===
Q: How often visit? A: Every 6 months.
Q: Whitening safe? A: Yes, completely safe.
Q: Child first visit? A: Age 2 or first tooth.
Q: Emergency? A: Yes, call +92-320-5018079 anytime.
Q: Implant duration? A: 3-6 months total.
Q: Painful? A: Latest anesthesia used, mostly pain-free.
"""

# ── Build RAG ─────────────────────────────────────────────────────────────────
print("Building knowledge base...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents([CLINIC_KNOWLEDGE])
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
print("Knowledge base ready!")

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI-powered dental clinic assistant for Taj Dentals website.
Answer using ONLY the provided context. No external knowledge. No hallucination.

LANGUAGE RULES:
- Detect language automatically: English, Urdu, Arabic, Roman Urdu
- Roman Urdu → respond in proper Roman Urdu script
- Urdu → respond in proper Urdu script
- English → respond in proper English script
- Always reply in the same language as the user

BEHAVIOR:
- Polite, professional, friendly
- Clear and concise answers
- Emergency → suggest calling clinic immediately
- Not in context → "Please contact the clinic for more information."

CONTEXT:
{context}"""

class ChatRequest(BaseModel):
    message: str
    history: list = []

class AppointmentRequest(BaseModel):
    name: str
    phone: str
    datetime: str
    service: str
    message: str = ""
    bookedAt: str

class FeedbackRequest(BaseModel):
    name: str
    rating: int
    service: str
    message: str
    bookedAt: str

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, req: ChatRequest):
    # ── Input Validation ──
    if not req.message.strip():
        return {"reply": "Please enter a message."}
    if len(req.message) > 500:
        return {"reply": "Message too long. Please keep it under 500 characters."}

    docs = retriever.invoke(req.message)
    context = "\n\n".join(d.page_content for d in docs)
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for m in req.history[-10:]:  # limit history to last 10 messages
        api_messages.append({"role": m["role"], "content": m["content"]})
    api_messages.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.3,
            max_tokens=512,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Sorry, an error occurred. Please try again."
    return {"reply": reply}

@app.post("/appointment")
async def save_appointment(req: AppointmentRequest):
    if SHEETS_URL:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(SHEETS_URL, json=req.dict())
        except Exception:
            pass
    return {"status": "success"}

@app.post("/feedback")
async def save_feedback(req: FeedbackRequest):
    if SHEETS_URL:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(SHEETS_URL, json=req.dict())
        except Exception:
            pass
    return {"status": "success"}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
