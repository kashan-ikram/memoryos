import streamlit as st 
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from supabase import create_client
import os, uuid, datetime, hashlib, json, tempfile, base64

# ── Setup ──
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
client = Groq()
client_db = chromadb.EphemeralClient()

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()

# ── Auth Functions ──
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register(username, email, password):
    try:
        users = load_users()
        if username in users:
            return False, "❌ Username already exists!"
        res = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"username": username}}
        })
        if res.user:
            users[username] = {
                "email": email,
                "password": hash_pw(password),
                "created": str(datetime.datetime.now())
            }
            save_users(users)
            return True, "ok"
        return False, "❌ Error aaya!"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def login(username, password):
    try:
        users = load_users()
        if username not in users:
            return False, "❌ Username nahi mila!"
        email = users[username]["email"]
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password,
        })
        if res.user:
            return True, "✅ Login ho gaya!"
        return False, "❌ Galat password!"
    except Exception as e:
        return False, "❌ Pehle email verify karo!"

def forgot_password(username):
    try:
        users = load_users()
        if username not in users:
            return False, "❌ Username nahi mila!"
        email = users[username]["email"]
        supabase.auth.reset_password_email(email)
        return True, f"✅ Reset link {email} pe bhej diya!"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

# ── Memory Functions — Supabase Permanent ──
def get_collection(username):
    try:
        return client_db.get_collection(f"memory_{username}")
    except:
        return client_db.create_collection(f"memory_{username}")

def remember(username, text, tags="manual", source="text"):
    try:
        # Supabase mein save karo — permanent
        supabase.table("memories").insert({
            "username": username,
            "content": text,
            "tags": f"{tags},{source}",
            "created_at": str(datetime.datetime.now())
        }).execute()
        supabase.table("episodes").insert({
            "username": username,
            "content": text,
            "tags": f"{tags},{source}",
            "created_at": str(datetime.datetime.now())
        }).execute()
        # ChromaDB mein bhi save karo — fast search ke liye
        col = get_collection(username)
        emb = model.encode([text]).tolist()
        col.add(documents=[text], embeddings=emb, ids=[str(uuid.uuid4())])
        return True
    except Exception as e:
        return False

def load_memories_to_chromadb(username):
    try:
        col = get_collection(username)
        existing = col.get()
        if len(existing["documents"]) > 0:
            return
        res = supabase.table("memories").select("content").eq("username", username).execute()
        if res.data:
            for item in res.data:
                emb = model.encode([item["content"]]).tolist()
                col.add(documents=[item["content"]], embeddings=emb, ids=[str(uuid.uuid4())])
    except:
        pass

def ask_ai(username, question):
    load_memories_to_chromadb(username)
    col = get_collection(username)
    emb = model.encode([question]).tolist()
    try:
        results = col.query(query_embeddings=emb, n_results=3)
        if not results["documents"][0]:
            return "Abhi koi memory nahi — Notes mein kuch add karo!"
        context = "\n".join(results["documents"][0])
    except:
        return "Abhi koi memory nahi — Notes mein kuch add karo!"

    prompt = f"""Tum MemoryOS ho — {username} ka personal AI assistant.
Memory se yeh mila:
{context}
Sawal: {question}
IMPORTANT INSTRUCTIONS:
- Hamesha Roman Urdu mein jawab do — Hindi ya English nahi
- Greeting mein hamesha "Assalamualaikum" use karo — "Namaste" bilkul nahi
- Friendly aur helpful raho"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_timeline(username):
    try:
        res = supabase.table("episodes").select("created_at,content,tags").eq("username", username).order("created_at", desc=True).limit(20).execute()
        return [(r["created_at"], r["content"], r["tags"]) for r in res.data]
    except:
        return []

def get_memories(username):
    try:
        res = supabase.table("memories").select("content").eq("username", username).execute()
        return [r["content"] for r in res.data]
    except:
        return []

# ── Audio Transcription ──
def transcribe_audio(audio_file):
    try:
        audio_bytes = audio_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".aac") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text"
            )
        os.unlink(tmp_path)
        return transcription, "detected"
    except Exception as e:
        return None, str(e)

# ── Image OCR ──
def extract_text_from_image(image_file):
    try:
        import pytesseract
        from PIL import Image
        import io
        img_bytes = image_file.read()
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        if text.strip():
            return text.strip()
        return "Image mein text nahi mila!"
    except Exception as e:
        return f"Error: {str(e)}"

# ── PAGE CONFIG ──
st.set_page_config(page_title="MemoryOS", page_icon="🧠", layout="wide")

# ── MEGA CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: #04020f !important; color: #e0d8ff !important; font-family: 'Rajdhani', sans-serif !important; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 1000px !important; margin: 0 auto !important; }

#particles-canvas { position: fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:0; }

.hero { text-align:center; padding:40px 20px 24px; position:relative; z-index:10; }
.hero-title { font-family:'Orbitron',monospace !important; font-size:clamp(32px,5vw,60px); font-weight:900;
    background:linear-gradient(135deg,#818cf8,#a78bfa,#60a5fa); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; letter-spacing:6px; filter:drop-shadow(0 0 30px rgba(139,92,246,0.5)); }
.hero-sub { font-size:12px; letter-spacing:6px; color:#6366f1; text-transform:uppercase; margin-top:6px; }
.hero-badge { display:inline-flex; align-items:center; gap:6px; margin-top:12px; padding:5px 16px;
    border:1px solid rgba(99,102,241,0.4); border-radius:20px; font-size:10px; letter-spacing:3px;
    color:#818cf8; background:rgba(99,102,241,0.08); }
.hero-badge::before { content:''; width:6px; height:6px; border-radius:50%; background:#4ade80;
    box-shadow:0 0 8px #4ade80; animation:blink 1.5s ease infinite; }

.login-card { background:rgba(15,8,40,0.9); border:1px solid rgba(139,92,246,0.3);
    border-radius:20px; padding:40px; max-width:440px; margin:0 auto;
    box-shadow:0 0 60px rgba(88,28,255,0.15); position:relative; z-index:10; }
.login-title { font-family:'Orbitron',monospace; font-size:14px; letter-spacing:4px;
    color:#a78bfa; text-align:center; margin-bottom:24px; }

.user-banner { background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
    border-radius:12px; padding:12px 20px; display:flex; align-items:center;
    justify-content:space-between; margin-bottom:20px; position:relative; z-index:10; }
.user-name { font-family:'Orbitron',monospace; font-size:12px; color:#a78bfa; letter-spacing:2px; }
.user-label { font-size:10px; color:#6366f1; letter-spacing:3px; }

.stats-row { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; position:relative; z-index:10; }
.stat-card { padding:16px; border-radius:12px; background:rgba(15,8,40,0.8);
    border:1px solid rgba(139,92,246,0.2); text-align:center; transition:all 0.3s; }
.stat-card:hover { border-color:rgba(139,92,246,0.5); box-shadow:0 0 20px rgba(88,28,255,0.15); }
.stat-num { font-family:'Orbitron',monospace; font-size:28px; font-weight:900;
    background:linear-gradient(135deg,#818cf8,#60a5fa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.stat-label { font-size:10px; letter-spacing:3px; color:#6366f1; text-transform:uppercase; margin-top:4px; }

.glass-card { background:rgba(15,8,40,0.8); border:1px solid rgba(139,92,246,0.2);
    border-radius:16px; padding:24px; margin-bottom:16px; backdrop-filter:blur(20px);
    box-shadow:0 0 40px rgba(88,28,255,0.08); position:relative; z-index:10; }

.input-tabs { display:flex; gap:8px; margin-bottom:16px; }
.input-tab { padding:8px 16px; border-radius:8px; border:1px solid rgba(99,102,241,0.3);
    background:rgba(99,102,241,0.06); color:#6366f1; font-family:'Orbitron',monospace;
    font-size:10px; letter-spacing:2px; cursor:pointer; transition:all 0.3s; }
.input-tab.active { background:rgba(99,102,241,0.2); border-color:#818cf8; color:#e0d8ff; }

.section-title { font-family:'Orbitron',monospace; font-size:11px; letter-spacing:4px;
    color:#6366f1; text-transform:uppercase; margin-bottom:16px; padding-bottom:10px;
    border-bottom:1px solid rgba(99,102,241,0.2); display:flex; align-items:center; gap:10px; }
.section-title::before { content:''; width:3px; height:16px;
    background:linear-gradient(#6366f1,#8b5cf6); border-radius:2px; box-shadow:0 0 8px #6366f1; }

.chat-msg { display:flex; gap:12px; margin-bottom:16px; }
.chat-msg.user-msg { flex-direction:row-reverse; }
.chat-avatar { width:36px; height:36px; border-radius:50%; display:flex;
    align-items:center; justify-content:center; font-size:16px; flex-shrink:0; }
.av-user { background:linear-gradient(135deg,#6366f1,#8b5cf6); box-shadow:0 0 12px rgba(99,102,241,0.5); }
.av-ai { background:linear-gradient(135deg,#1e40af,#3b82f6); box-shadow:0 0 12px rgba(59,130,246,0.5); }
.chat-bubble { max-width:70%; padding:12px 18px; border-radius:16px; font-size:15px; line-height:1.6; }
.bubble-user { background:linear-gradient(135deg,rgba(99,102,241,0.3),rgba(139,92,246,0.3));
    border:1px solid rgba(139,92,246,0.4); color:#e0d8ff; border-bottom-right-radius:4px; }
.bubble-ai { background:rgba(15,23,42,0.8); border:1px solid rgba(59,130,246,0.3);
    color:#bfdbfe; border-bottom-left-radius:4px; }

.memory-card { padding:12px 16px; border-radius:10px; background:rgba(99,102,241,0.06);
    border:1px solid rgba(99,102,241,0.2); border-left:3px solid #6366f1;
    margin-bottom:8px; font-size:14px; color:#c4b8ff; transition:all 0.3s; }
.memory-card:hover { background:rgba(99,102,241,0.12); transform:translateX(4px); }

.source-badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px;
    letter-spacing:1px; margin-left:6px; }
.source-text { background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3); color:#818cf8; }
.source-audio { background:rgba(74,222,128,0.1); border:1px solid rgba(74,222,128,0.3); color:#4ade80; }
.source-image { background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.3); color:#fbbf24; }

.timeline-item { display:flex; gap:16px; margin-bottom:16px; }
.tl-dot { width:12px; height:12px; border-radius:50%; background:linear-gradient(#6366f1,#8b5cf6);
    box-shadow:0 0 10px rgba(99,102,241,0.6); flex-shrink:0; margin-top:4px; }
.tl-time { font-family:'Orbitron',monospace; font-size:10px; color:#6366f1; letter-spacing:2px; margin-bottom:4px; }
.tl-text { font-size:14px; color:#c4b8ff; line-height:1.5; }
.tl-tag { display:inline-block; padding:2px 8px; border-radius:4px; background:rgba(99,102,241,0.15);
    border:1px solid rgba(99,102,241,0.3); font-size:10px; color:#818cf8; margin-top:4px; }

.guide-step { display:flex; align-items:flex-start; gap:16px; padding:16px;
    border-radius:12px; background:rgba(99,102,241,0.05); border:1px solid rgba(99,102,241,0.15);
    margin-bottom:12px; transition:all 0.3s; }
.guide-step:hover { background:rgba(99,102,241,0.1); transform:translateX(4px); }
.step-num { min-width:36px; height:36px; border-radius:50%; background:linear-gradient(135deg,#6366f1,#8b5cf6);
    display:flex; align-items:center; justify-content:center; font-family:'Orbitron',monospace;
    font-size:11px; font-weight:700; color:white; box-shadow:0 0 16px rgba(99,102,241,0.5); }
.step-title { font-family:'Orbitron',monospace; font-size:11px; color:#a78bfa; letter-spacing:2px; margin-bottom:4px; }
.step-desc { font-size:14px; color:#c4b8ff; line-height:1.6; }

.forgot-box { background:rgba(99,102,241,0.05); border:1px solid rgba(99,102,241,0.2);
    border-radius:12px; padding:16px; margin-top:12px; }

.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background:rgba(15,8,40,0.9) !important; border:1px solid rgba(139,92,246,0.3) !important;
    border-radius:10px !important; color:#e0d8ff !important; font-family:'Rajdhani',sans-serif !important; font-size:15px !important; }
.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color:#8b5cf6 !important; box-shadow:0 0 20px rgba(139,92,246,0.2) !important; }
.stTextInput label, .stTextArea label { color:#6366f1 !important; font-family:'Orbitron',monospace !important;
    font-size:10px !important; letter-spacing:3px !important; text-transform:uppercase !important; }

.stButton>button { background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border:none !important; border-radius:10px !important; color:white !important;
    font-family:'Orbitron',monospace !important; font-size:11px !important;
    letter-spacing:2px !important; padding:12px 24px !important; transition:all 0.3s !important;
    box-shadow:0 0 20px rgba(99,102,241,0.3) !important; width:100% !important; }
.stButton>button:hover { transform:translateY(-2px) !important; box-shadow:0 0 40px rgba(99,102,241,0.5) !important; }

.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid rgba(99,102,241,0.2) !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; border:1px solid transparent !important;
    border-radius:8px 8px 0 0 !important; color:#6366f1 !important; font-family:'Orbitron',monospace !important;
    font-size:10px !important; letter-spacing:2px !important; transition:all 0.3s !important; }
.stTabs [aria-selected="true"] { background:rgba(99,102,241,0.15) !important;
    border-color:rgba(99,102,241,0.4) !important; color:#a78bfa !important; }
.stTabs [data-baseweb="tab-panel"] { padding:20px 0 !important; }

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:rgba(99,102,241,0.4); border-radius:2px; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
</style>

<canvas id="particles-canvas"></canvas>
<script>
const c=document.getElementById('particles-canvas');
const x=c.getContext('2d');
c.width=window.innerWidth; c.height=window.innerHeight;
const p=Array.from({length:70},()=>({
    x:Math.random()*c.width, y:Math.random()*c.height,
    vx:(Math.random()-.5)*.4, vy:(Math.random()-.5)*.4,
    r:Math.random()*1.5+.5, a:Math.random()*.4+.1,
    col:Math.random()>.5?'139,92,246':'96,165,250'
}));
function draw(){
    x.clearRect(0,0,c.width,c.height);
    p.forEach(d=>{
        d.x+=d.vx; d.y+=d.vy;
        if(d.x<0||d.x>c.width)d.vx*=-1;
        if(d.y<0||d.y>c.height)d.vy*=-1;
        x.beginPath(); x.arc(d.x,d.y,d.r,0,Math.PI*2);
        x.fillStyle=`rgba(${d.col},${d.a})`; x.fill();
    });
    for(let i=0;i<p.length;i++) for(let j=i+1;j<p.length;j++){
        const dx=p[i].x-p[j].x, dy=p[i].y-p[j].y, dist=Math.sqrt(dx*dx+dy*dy);
        if(dist<90){x.beginPath();x.moveTo(p[i].x,p[i].y);x.lineTo(p[j].x,p[j].y);
            x.strokeStyle=`rgba(139,92,246,${.07*(1-dist/90)})`;x.lineWidth=.5;x.stroke();}
    }
    requestAnimationFrame(draw);
}
draw();
window.addEventListener('resize',()=>{c.width=window.innerWidth;c.height=window.innerHeight;});
</script>
""", unsafe_allow_html=True)

# ── SESSION STATE ──
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_forgot" not in st.session_state:
    st.session_state.show_forgot = False

# ════════════════════════════════════════════════════════
# LOGIN PAGE
# ════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">MemoryOS</div>
        <div class="hero-sub">Personal AI Second Brain</div>
        <div class="hero-badge">SECURE LOGIN</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">APNA ACCOUNT</div>', unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔐  LOGIN", "✨  REGISTER"])

        with tab_login:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            l_user = st.text_input("USERNAME", key="l_user", placeholder="apna username")
            l_pass = st.text_input("PASSWORD", key="l_pass", placeholder="apna password", type="password")

            if st.button("🚀  LOGIN KARO", key="login_btn"):
                if l_user and l_pass:
                    ok, msg = login(l_user, l_pass)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username = l_user
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Username aur password dono likho!")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("🔑 Password Bhool Gaye?", key="forgot_toggle"):
                st.session_state.show_forgot = not st.session_state.show_forgot

            if st.session_state.show_forgot:
                st.markdown('<div class="forgot-box">', unsafe_allow_html=True)
                f_user = st.text_input("USERNAME LIKHO", key="f_user", placeholder="apna username")
                if st.button("📧 Reset Email Bhejo", key="forgot_btn"):
                    if f_user:
                        ok, msg = forgot_password(f_user)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Username likho!")
                st.markdown('</div>', unsafe_allow_html=True)

        with tab_register:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            r_user = st.text_input("USERNAME CHOOSE KARO", key="r_user", placeholder="naya username")
            r_email = st.text_input("GMAIL ADDRESS", key="r_email", placeholder="example@gmail.com")
            r_pass = st.text_input("PASSWORD RAKHO", key="r_pass", placeholder="strong password", type="password")
            r_pass2 = st.text_input("PASSWORD CONFIRM KARO", key="r_pass2", placeholder="dobara likho", type="password")

            if st.button("✨  ACCOUNT BANAO", key="reg_btn"):
                if r_user and r_email and r_pass and r_pass2:
                    if r_pass != r_pass2:
                        st.error("❌ Passwords match nahi kar rahe!")
                    elif len(r_pass) < 6:
                        st.error("❌ Password kam se kam 6 characters ka ho!")
                    elif "@" not in r_email:
                        st.error("❌ Sahi email likho!")
                    else:
                        ok, msg = register(r_user, r_email, r_pass)
                        if ok:
                            st.info(f"📧 {r_email} pe verification email bheja gaya! Email kholo — link click karo — phir wapas aao aur login karo!")
                        else:
                            st.error(msg)
                else:
                    st.warning("Sab fields bharo!")

        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════
else:
    username = st.session_state.username

    st.markdown("""
    <div class="hero">
        <div class="hero-title">MemoryOS</div>
        <div class="hero-sub">Personal AI Second Brain</div>
        <div class="hero-badge">SYSTEM ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class="user-banner">
            <div>
                <div class="user-label">LOGGED IN AS</div>
                <div class="user-name">👤 {username.upper()}</div>
            </div>
            <div style="font-size:11px; color:#4ade80; letter-spacing:2px;">🔒 PRIVATE MEMORY</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🚪 LOGOUT"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.messages = []
            st.rerun()

    all_mems = get_memories(username)
    tl_data = get_timeline(username)
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-num">{len(all_mems)}</div>
            <div class="stat-label">Memories</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{len(tl_data)}</div>
            <div class="stat-label">Episodes</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{len(st.session_state.messages)}</div>
            <div class="stat-label">Messages</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["💬  CHAT", "📝  NOTES", "📅  TIMELINE", "📖  GUIDE"])

    # ── CHAT ──
    with tab1:
        st.markdown('<div class="section-title">AI SE POOCHHO</div>', unsafe_allow_html=True)

        if st.session_state.messages:
            chat_html = '<div class="glass-card">'
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_html += f'''<div class="chat-msg user-msg">
                        <div class="chat-avatar av-user">👤</div>
                        <div class="chat-bubble bubble-user">{msg["content"]}</div>
                    </div>'''
                else:
                    chat_html += f'''<div class="chat-msg">
                        <div class="chat-avatar av-ai">🧠</div>
                        <div class="chat-bubble bubble-ai">{msg["content"]}</div>
                    </div>'''
            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;padding:40px;">
                <div style="font-size:48px;animation:float 3s ease infinite;">🧠</div>
                <div style="font-family:'Orbitron',monospace;font-size:13px;color:#6366f1;letter-spacing:3px;margin-top:12px;">
                    WELCOME {username.upper()}
                </div>
                <div style="font-size:14px;color:#6b7280;margin-top:8px;">
                    Tumhari private memory ready hai — kuch poochho!
                </div>
            </div>
            """, unsafe_allow_html=True)

        user_input = st.text_input("", placeholder="Kuch poochho...", key="chat_in", label_visibility="collapsed")
        col_a, col_b = st.columns([3, 1])
        with col_b:
            send = st.button("SEND →", key="send")

        if send and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Soch raha hun..."):
                resp = ask_ai(username, user_input)
            st.session_state.messages.append({"role": "assistant", "content": resp})
            st.rerun()

    # ── NOTES — TEXT + AUDIO + IMAGE ──
    with tab2:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown('<div class="section-title">NAYA NOTE ADD KARO</div>', unsafe_allow_html=True)

            input_type = st.radio("", ["📝 Text", "🎙️ Audio", "🖼️ Image"],
                                  horizontal=True, label_visibility="collapsed")

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            # ── TEXT INPUT ──
            if input_type == "📝 Text":
                note = st.text_area("NOTE", placeholder="Kuch bhi likho — idea, meeting, task...", height=120)
                tags = st.text_input("TAGS", placeholder="work, idea, personal")
                if st.button("💾  MEMORY MEIN SAVE KARO", key="save_text"):
                    if note.strip():
                        remember(username, note, tags=tags, source="text")
                        st.success("✅ Yaad ho gaya!")
                        st.rerun()
                    else:
                        st.warning("Pehle kuch likho!")

            # ── AUDIO INPUT ──
            elif input_type == "🎙️ Audio":
                st.markdown('<div style="color:#4ade80;font-size:13px;margin-bottom:12px;">🎙️ Audio file upload karo ya record karo</div>', unsafe_allow_html=True)

                audio_source = st.radio("", ["📁 File Upload", "🎤 Record Karo"],
                                       horizontal=True, label_visibility="collapsed")

                if audio_source == "📁 File Upload":
                    audio_file = st.file_uploader("AUDIO FILE (Max 5MB)", 
                               type=["mp3", "wav", "aac", "m4a", "ogg"],
                               label_visibility="visible")
                    tags = st.text_input("TAGS", placeholder="voice, meeting", key="audio_tags")
                    if st.button("🎙️ TRANSCRIBE + SAVE", key="save_audio"):
                        if audio_file:
                            with st.spinner("Audio sun raha hun..."):
                                text, lang = transcribe_audio(audio_file)
                            if text:
                                st.info(f"📝 Transcribed ({lang}): {text}")
                                remember(username, text, tags=tags, source="audio")
                                st.success("✅ Voice memory save ho gayi!")
                                st.rerun()
                            else:
                                st.error(f"❌ Error: {lang}")
                        else:
                            st.warning("Pehle audio file upload karo!")

                else:
                    st.markdown("""
                    <div style="background:rgba(74,222,128,0.05);border:1px solid rgba(74,222,128,0.2);
                    border-radius:10px;padding:16px;text-align:center;">
                        <div style="font-size:32px">🎤</div>
                        <div style="color:#4ade80;font-size:13px;margin-top:8px;">
                            Phone pe Voice Memo record karo — phir File Upload se upload karo!
                        </div>
                        <div style="color:#6b7280;font-size:11px;margin-top:4px;">
                            WhatsApp audio bhi kaam karta hai ✅
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── IMAGE INPUT ──
            elif input_type == "🖼️ Image":
                st.markdown('<div style="color:#fbbf24;font-size:13px;margin-bottom:12px;">🖼️ Image upload karo — AI text extract karega</div>', unsafe_allow_html=True)

                image_file = st.file_uploader("IMAGE FILE (Max 5MB)", type=["jpg", "jpeg", "png", "webp"],
                                              label_visibility="visible")
                tags = st.text_input("TAGS", placeholder="screenshot, document", key="img_tags")

                if image_file:
                    st.image(image_file, caption="Uploaded Image", use_column_width=True)

                if st.button("🖼️ EXTRACT + SAVE", key="save_image"):
                    if image_file:
                        with st.spinner("Image analyze kar raha hun..."):
                            extracted = extract_text_from_image(image_file)
                        st.info(f"📝 Extracted: {extracted[:200]}...")
                        remember(username, extracted, tags=tags, source="image")
                        st.success("✅ Image memory save ho gayi!")
                        st.rerun()
                    else:
                        st.warning("Pehle image upload karo!")

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-title">RECENT MEMORIES</div>', unsafe_allow_html=True)
            mems = get_memories(username)
            if mems:
                html = ""
                for doc in list(reversed(mems))[:8]:
                    html += f'<div class="memory-card">{doc[:80]}{"..." if len(doc) > 80 else ""}</div>'
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.markdown('<div class="glass-card" style="text-align:center;padding:30px;"><div style="font-size:32px">📭</div><div style="color:#6b7280;margin-top:8px;">Abhi koi memory nahi</div></div>', unsafe_allow_html=True)

    # ── TIMELINE ──
    with tab3:
        st.markdown('<div class="section-title">TUMHARI TIMELINE</div>', unsafe_allow_html=True)
        tl = get_timeline(username)
        if tl:
            html = '<div class="glass-card">'
            for ts, content, tag in tl:
                tag_html = f'<span class="tl-tag">{tag}</span>' if tag else ""
                html += f'''<div class="timeline-item">
                    <div class="tl-dot" style="margin-top:4px;flex-shrink:0;"></div>
                    <div>
                        <div class="tl-time">{str(ts)[:16]}</div>
                        <div class="tl-text">{content[:100]}{"..." if len(content) > 100 else ""}</div>
                        {tag_html}
                    </div>
                </div>'''
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass-card" style="text-align:center;padding:40px;"><div style="font-size:40px">📅</div><div style="color:#6b7280;margin-top:8px;">Abhi koi timeline nahi</div></div>', unsafe_allow_html=True)

    # ── GUIDE ──
    with tab4:
        st.markdown('<div class="section-title">USER GUIDE</div>', unsafe_allow_html=True)
        steps = [
            ("01", "REGISTER KARO", "✨ Register tab mein jao — username, Gmail aur password rakho!"),
            ("02", "EMAIL VERIFY KARO", "📧 Gmail check karo — verification link click karo — account ready!"),
            ("03", "TEXT NOTE ADD KARO", "📝 Notes > Text — kuch bhi likho — memory mein save ho jayega!"),
            ("04", "VOICE NOTE ADD KARO", "🎙️ Notes > Audio — audio file upload karo — AI transcribe karega!"),
            ("05", "IMAGE ADD KARO", "🖼️ Notes > Image — screenshot ya document upload karo — AI text extract karega!"),
            ("06", "AI SE POOCHHO", "💬 Chat tab — koi bhi sawal poochho — AI memory se jawab dega!"),
            ("07", "TIMELINE DEKHO", "📅 Timeline tab — sab memories date ke saath dikhenge!"),
            ("08", "PASSWORD RESET", "🔑 Login page > Password Bhool Gaye — email se reset karo!"),
        ]
        html = '<div class="glass-card">'
        for num, title, desc in steps:
            html += f'''<div class="guide-step">
                <div class="step-num">{num}</div>
                <div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>'''
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
