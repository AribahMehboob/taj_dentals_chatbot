import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv('dental')
if not api_key:
    raise RuntimeError("Missing 'dental' API key in .env")
client = Groq(api_key=api_key)

# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Serve static files ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Logo route (optional – already covered by static mount) ──────────────────
@app.get("/images/logo.png")
async def get_logo():
    return FileResponse("static/images/logo.png")

# ── Clinic Knowledge ──────────────────────────────────────────────────────────


#  ── Clinic Knowledge ───────────────────────────────────────────────────────────
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
Address : Rawalpindi Road, Dadyal, AJK
clinic is located at Rawalpindi Road, Dadyal, AJK
Landmark: Near THQ Dadyal. Free parking available.

# === CEO ===
# CEO of taj dentals is  Mirza Muhammand Mehboob 
# who is aloso CEO of Mughal Construction Group
# don't mixup ceo and doctors 



=== CONTACT ===
Phone: +92-320-5018079
WhatsApp: +92-320-5018079
Email: tajdentals01@gmail.com

=== WORKING HOURS ===
Clinic is open every day: 9:00 AM - 7:00 PM
Surgeon is unavailable on Fridays. But clinic is open on friday
Public Holidays: Closed (Emergency contact available)

=== DOCTORS ===
Dr. Iqra Mehboob -Dental Surgen, BDS RDS , 5 years experience
Mr. Qamar  - Dental Hygenist and Dental Techician, 7 years experience
Mr. Ali-  Dental Assistant, 3 years experience

=== FEES & PAYMENT ===
Consultation Fee: Rs. 1,000
scaling + Polishing: Rs. 5,500 - 8,000
Tooth Filling: Rs. 2,000 - 3,000
Laser Filing: 3,000 - 4,500
Root Canal Treatment: Rs. 5,000 - 8,000
Tooth Extraction (simple): Rs. 1,000 - 3,500
Tooth Extraction (surgical): Rs. 7,000 - 10,000
Teeth Whitening: Rs. 25,000 - 30,000
Dental Implant: Rs. 100,000 - 150,000
Dental Braces: Rs. 50,000 - 80,000
Crown (Porcelain): Rs. 10,000 - 18,000
Payment: Cash, EasyPaisa, JazzCash, Cards, Bank Transfer.

=== SPECIAL OFFERS ===
Free first consultation for new patients
Family package: 10% discount
Senior citizens: 15% discount
Student discount: 10% off with valid ID

=== FAQs ===
Q: How often visit? A: Every 6 months.
Q: Whitening safe? A: Yes, completely safe.
Q: Child first visit? A: Age 2 or first tooth.
Q: Emergency? A: Yes, call +92-300-1234567 anytime.
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

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI-powered dental clinic assistant for Taj Dentals website.
Answer using ONLY the provided context. No external knowledge. No hallucination.

LANGUAGE RULES:
- Detect language automatically: English, Urdu, Arabic, Roman Urdu
- Roman Urdu → respond in proper Roman Urdu script
- Urdu → respond in proper Urdu script
- english → respond in proper english script
- Always reply in the same language as the user everytime o every query

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

@app.post("/chat")
async def chat(req: ChatRequest):
    docs = retriever.invoke(req.message)
    context = "\n\n".join(d.page_content for d in docs)
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for m in req.history:
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
        reply = f"Sorry, an error occurred: {str(e)}"
    return {"reply": reply}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
     

    import os
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    # port = int(os.getenv("PORT", 7860))



    # uvicorn.run(app, host="0.0.0.0", port=7860)

