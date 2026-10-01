from pydantic import BaseModel, Field
import os
import json
from datetime import datetime
import streamlit as st
def get_secret(key):
    """Διαβάζει από st.secrets (Cloud) ή get_secret (τοπικά)."""
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from pollinations import Pollinations
from gtts import gTTS
from io import BytesIO
from faster_whisper import WhisperModel
import tempfile
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware, PIIMiddleware

load_dotenv()

MEMORY_FILE = "memory.json"

# ===== ΡΥΘΜΙΣΗ ΣΕΛΙΔΑΣ =====

st.set_page_config(
    page_title="Ο AI Agent μου",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== ΘΕΜΑΤΑ =====

# ===== SMART SUGGESTIONS MODEL =====

class AgentResponse(BaseModel):
    """Δομημένη απάντηση με προτάσεις."""
    answer: str = Field(description="Η απάντηση του agent")
    suggestions: list[str] = Field(
        description="3 προτεινόμενες επόμενες ερωτήσεις",
        default_factory=list
    )

THEMES = {
    "cyberpunk": {
        "name": "🌃 Cyberpunk",
        "bg": "#0B1021",
        "card": "#141A31",
        "text": "#E6F1FF",
        "primary": "#00FF9C",
        "sub": "#A0AEC0",
        "border": "#00FF9C"
    },
    "matrix": {
        "name": "💚 Matrix",
        "bg": "#000000",
        "card": "#0A1A0A",
        "text": "#00FF00",
        "primary": "#00FF00",
        "sub": "#00AA00",
        "border": "#00FF00"
    },
    "sunset": {
        "name": "🌅 Sunset",
        "bg": "#1A0F0A",
        "card": "#2A1A0F",
        "text": "#FFE4C4",
        "primary": "#FF6B35",
        "sub": "#C09070",
        "border": "#FF6B35"
    },
    "ocean": {
        "name": "🌊 Ocean",
        "bg": "#0A1628",
        "card": "#152A45",
        "text": "#E0F4FF",
        "primary": "#00BFFF",
        "sub": "#7FB0D0",
        "border": "#00BFFF"
    },
    "purple": {
        "name": "💜 Neon Purple",
        "bg": "#0F0A1E",
        "card": "#1A1030",
        "text": "#E8D5FF",
        "primary": "#B026FF",
        "sub": "#A080C0",
        "border": "#B026FF"
    }
}

# Αρχικοποίηση θέματος
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "cyberpunk"

theme = THEMES[st.session_state.theme_name]

st.markdown(f"""
<style>
    /* Κύριο φόντο */
    .stApp {{
        background: {theme['bg']};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {theme['card']};
        border-right: 1px solid {theme['border']};
    }}
    
    /* Κείμενο */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {theme['text']};
    }}
    
    /* Μηνύματα */
    .stChatMessage {{
        background: {theme['card']};
        border-left: 3px solid {theme['primary']};
        border-radius: 12px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        animation: fadeIn 0.4s ease-in-out;
        transition: all 0.3s ease;
    }}
    
    .stChatMessage:hover {{
        box-shadow: 0 0 15px {theme['primary']}40;
    }}
    
    /* Κουμπιά */
    .stButton button {{
        background: linear-gradient(135deg, {theme['primary']}, {theme['primary']}CC);
        color: {theme['bg']};
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px {theme['primary']}30;
    }}
    
    .stButton button:hover {{
        box-shadow: 0 0 25px {theme['primary']}80;
        transform: translateY(-2px);
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {theme['bg']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {theme['primary']};
        border-radius: 5px;
    }}
    
    /* Chat input */
    .stChatInput {{
        border: 2px solid {theme['primary']} !important;
        border-radius: 12px;
        transition: all 0.3s ease;
    }}
    
    .stChatInput:focus-within {{
        box-shadow: 0 0 20px {theme['primary']}50;
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Header */
    .main-header {{
        animation: fadeIn 0.6s ease-in-out;
        box-shadow: 0 0 20px {theme['primary']}20;
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {theme['primary']};
    }}
    
    /* Smooth scroll */
    html {{
        scroll-behavior: smooth;
    }}
</style>
""", unsafe_allow_html=True)


# ===== CUSTOM CSS =====

st.markdown("""
<style>
    /* Fade-in animation για μηνύματα */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.4s ease-in-out;
        border-left: 3px solid #00FF9C;
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        border-left: 3px solid #00FF9C;
        box-shadow: 0 0 15px rgba(0, 255, 156, 0.3);
    }
    
    /* Glow στα κουμπιά */
    .stButton button {
        background: linear-gradient(135deg, #00FF9C, #00CC7A);
        color: #0B1021;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 255, 156, 0.3);
    }
    
    .stButton button:hover {
        box-shadow: 0 0 25px rgba(0, 255, 156, 0.8);
        transform: translateY(-2px);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0B1021;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00FF9C;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00CC7A;
    }
    
    /* Chat input glow */
    .stChatInput {
        border: 2px solid #00FF9C !important;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stChatInput:focus-within {
        box-shadow: 0 0 20px rgba(0, 255, 156, 0.5);
    }
    
    /* Header glow */
    .main-header {
        animation: fadeIn 0.6s ease-in-out;
        box-shadow: 0 0 20px rgba(0, 255, 156, 0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #080C1A;
        border-right: 1px solid #00FF9C;
    }
    
    /* Smooth scroll */
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: #141A31; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #00FF9C; margin-bottom: 1rem;">
    <h1 style="color: #00FF9C; margin: 0;">🤖 Ο AI Agent μου</h1>
    <p style="color: #E6F1FF; margin: 0;">Ο προσωπικός σου βοηθός με μνήμη 🧠</p>
</div>
""", unsafe_allow_html=True)

# ===== CSS =====

# ===== ΘΕΜΑ (LIGHT / DARK) =====

# st.markdown(f"""
# <style>...</style>
# """, unsafe_allow_html=True)

# ===== ΕΡΓΑΛΕΙΑ =====

@tool
def calculate(expression: str) -> str:
    """Υπολογίζει μια μαθηματική παράσταση. Π.χ. '15 * 23 + 100'"""
    try:
        result = eval(expression)
        return f"Αποτέλεσμα: {result}"
    except Exception as e:
        return f"Σφάλμα: {e}"

@tool
def get_time() -> str:
    """Επιστρέφει την τρέχουσα ώρα και ημερομηνία."""
    now = datetime.now()
    return now.strftime("Είναι %H:%M, %d/%m/%Y")

@tool
def word_count(text: str) -> str:
    """Μετράει πόσες λέξεις έχει ένα κείμενο."""
    words = len(text.split())
    return f"Το κείμενο έχει {words} λέξεις."

tavily = TavilySearch(max_results=3)
@tool
def generate_image(prompt: str) -> str:
    """Δημιουργεί μια εικόνα με βάση μια περιγραφή κειμένου. Χρησιμοποίησέ το όταν ο χρήστης ζητά να σχεδιαστεί κάτι."""
    try:
        client = Pollinations()
        image_url = client.generate_image(prompt)
        return f"Η εικόνα δημιουργήθηκε: {image_url}"
    except Exception as e:
        return f"Η δημιουργία απέτυχε: {e}"
python_repl = PythonREPLTool()
# ===== SUBAGENT: ΕΡΕΥΝΗΤΗΣ =====

research_agent = create_agent(
    model=(
        ChatOllama(model="qwen3:8b", temperature=0.7)
        if get_secret("USE_OLLAMA") == "true"
        else ChatOpenRouter(
            model="qwen/qwen-2.5-72b-instruct:free",
            api_key=get_secret("OPENROUTER_API_KEY")
        )
    ),
    tools=[tavily],
    system_prompt="Είσαι ένας ειδικός ερευνητής. Ψάχνεις στο internet με το TavilySearch και επιστρέφεις ΜΟΝΟ τα βασικά συμπεράσματα, συνοπτικά, χωρίς περιττά λόγια."
)

@tool
def delegate_research(query: str) -> str:
    """Ανάθεσε μια ερευνητική εργασία στον ειδικό ερευνητή. Χρησιμοποίησέ το όταν ο χρήστης χρειάζεται τρέχουσες πληροφορίες ή δεδομένα σε πραγματικό χρόνο."""
    result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content
# ===== ΜΝΗΜΗ =====

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_memory(messages):
    simple = []
    for m in messages:
        if isinstance(m, dict):
            simple.append({"role": m["role"], "content": m["content"]})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(simple, f, ensure_ascii=False, indent=2)

def generate_suggestions(answer: str, context: list) -> list:
    """Δημιουργεί 3 προτεινόμενες ερωτήσεις με βάση την απάντηση."""
    try:
        model_with_structure = model.with_structured_output(AgentResponse)
        
        prompt = f"""Με βάση αυτή την απάντηση:
        
"{answer}"

Πρότεινε 3 σύντομες, σχετικές ερωτήσεις που μπορεί να κάνει ο χρήστης στη συνέχεια.
Κάθε ερώτηση να είναι μέχρι 8 λέξεις, στα ελληνικά."""
        
        result = model_with_structure.invoke(prompt)
        return result.suggestions[:3]
    except Exception as e:
        print(f"Suggestions Error: {e}")
        return []
def export_chat_json(messages):
    """Εξάγει τη συνομιλία σε JSON."""
    data = {
        "exported_at": datetime.now().isoformat(),
        "total_messages": len(messages),
        "messages": messages
    }
    return json.dumps(data, ensure_ascii=False, indent=2)

def export_chat_txt(messages):
    """Εξάγει τη συνομιλία σε TXT."""
    lines = []
    for m in messages:
        role = "👤 Εσύ" if m.get("role") == "user" else "🤖 Agent"
        lines.append(f"{role}: {m.get('content', '')}")
        lines.append("")
    return "\n".join(lines)

@st.cache_resource
def load_whisper_model():
    """Φορτώνει το μοντέλο Whisper (μία φορά)."""
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe_audio(audio_file):
    """Μετατρέπει αρχείο ήχου σε κείμενο."""
    model = load_whisper_model()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    
    try:
        segments, info = model.transcribe(
            tmp_path,
            language="el",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join([seg.text for seg in segments])
        return text.strip()
    finally:
        os.unlink(tmp_path)

def generate_tts_gtts(text: str) -> bytes:
    """Δημιουργεί MP3 ήχο από κείμενο με gTTS (πιο σταθερό για ελληνικά)."""
    try:
        tts = gTTS(text=text, lang='el', slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        return f"ΣΦΑΛΜΑ: {e}".encode()
@tool
def generate_image(prompt: str) -> str:
    """Δημιουργεί μια εικόνα με βάση μια περιγραφή κειμένου."""
    try:
        client = Pollinations()  # χρησιμοποιεί το import από πάνω
        image_url = client.generate_image(prompt)
        return f"Η εικόνα δημιουργήθηκε: {image_url}"
    except Exception as e:
        return f"Η δημιουργία απέτυχε: {e}"
@tool
def enhance_prompt(prompt: str) -> str:
    """Βελτιώνει μια περιγραφή εικόνας προσθέτοντας λεπτομέρειες φωτισμού, στυλ και σύνθεσης. Χρησιμοποίησέ το ΠΡΙΝ από το generate_image."""
    try:
        enhancer_model = (
            ChatOllama(model="qwen3:8b", temperature=0.7)
            if get_secret("USE_OLLAMA") == "true"
            else ChatOpenRouter(
                model="qwen/qwen-2.5-72b-instruct:free",
                api_key=get_secret("OPENROUTER_API_KEY")
            )
        )
        result = enhancer_model.invoke(
            f"Πάρε αυτή την απλή περιγραφή και μετάτρεψέ την σε μια πλούσια, "
            f"κινηματογραφική περιγραφή για μοντέλο δημιουργίας εικόνας. "
            f"Πρόσθεσε φωτισμό, στυλ, σύνθεση, χρώματα. "
            f"Απάντησε ΜΟΝΟ με την βελτιωμένη περιγραφή στα αγγλικά.\n\n"
            f"Αρχική: {prompt}"
        )
        return result.content
    except Exception as e:
        return f"Η βελτίωση απέτυχε: {e}. Χρησιμοποίησε το αρχικό prompt."

# ===== ΜΟΝΤΕΛΟ & AGENT =====

@st.cache_resource
def get_agent():
    if get_secret("USE_OLLAMA") == "true":
        model = ChatOllama(
            model="qwen3:8b",
            temperature=0.7,
        )
    else:
        model = ChatOpenRouter(
            model="meta-llama/llama-3.3-70b-instruct:free",
            api_key=get_secret("OPENROUTER_API_KEY")
        )
    
    return create_agent(
        model=model,
        tools=[calculate, get_time, word_count, tavily, delegate_research, python_repl, generate_image, enhance_prompt],
        system_prompt="Απάντα ΠΑΝΤΑ στα Ελληνικά, ανεξάρτητα από τη γλώσσα της ερώτησης. Είσαι ένας εξυπηρετικός βοηθός. Ο χρήστης είναι ο άνθρωπος που σου μιλάει — εσύ ΔΕΝ είσαι ο χρήστης. Χρησιμοποίησε τα εργαλεία όταν χρειάζεται. Πάντα στο τέλος της  απάντησής σου, πρότεινε 3 σχετικές επόμενες ερωτήσεις που μπορεί να κάνει ο χρήστης. Για δημιουργία εικόνας: ΠΡΩΤΑ κάλεσε το enhance_prompt, ΜΕΤΑ πέρασε το αποτέλεσμα στο generate_image.",
        middleware=[
            PIIMiddleware("email", strategy="redact"),
            PIIMiddleware("credit_card", strategy="mask"),
            PIIMiddleware("url", strategy="redact"),
                        SummarizationMiddleware(
                model=(
                    "ollama:qwen3:8b"
                    if get_secret("USE_OLLAMA") == "true"
                    else "openrouter:meta-llama/llama-3.3-70b-instruct:free"
                ),
                trigger=("tokens", 4000),
                keep=("messages", 20)
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "python_repl": True,
                },
            ),
        ],
        checkpointer=MemorySaver()
    )

agent = get_agent()

# ===== HEADER =====

st.markdown("""
<div class="main-header">
    <h1>🤖 Ο AI Agent μου</h1>
    <p style="color: #718096; margin: 0;">Ο προσωπικός σου βοηθός με μνήμη 🧠</p>
</div>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====

with st.sidebar:
    # Theme Switcher
    st.markdown("### 🎨 Θέμα")
    selected_theme = st.selectbox(
        "Διάλεξε θέμα:",
        options=list(THEMES.keys()),
        format_func=lambda x: THEMES[x]["name"],
        index=list(THEMES.keys()).index(st.session_state.theme_name),
        label_visibility="collapsed"
    )
    
    if selected_theme != st.session_state.theme_name:
        st.session_state.theme_name = selected_theme
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ⚡ AI Agent")
    st.markdown("---")
    
    enable_tts = st.checkbox("🔊 Ενεργοποίηση Φωνής (TTS)", value=False)
    st.markdown("---")

    # Στατιστικά
    st.markdown("#### 📊 Στατιστικά")
    mem = load_memory()
    total = len(mem)
    user_msgs = len([m for m in mem if m.get("role") == "user"])
    agent_msgs = len([m for m in mem if m.get("role") == "assistant"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👤 Εσύ", user_msgs)
    with col2:
        st.metric("🤖 Agent", agent_msgs)
    
    st.metric("💾 Σύνολο", total)
    st.markdown("---")
    
    # Quick Actions
    st.markdown("#### ⚡ Γρήγορες Ενέργειες")
    
    if st.button("🧮 Υπολόγισε", use_container_width=True):
        st.session_state.quick_prompt = "Πόσο κάνει 144 / 12;"
    
    if st.button("🕐 Ώρα", use_container_width=True):
        st.session_state.quick_prompt = "Τι ώρα είναι;"
    
    if st.button("🌐 Νέα AI", use_container_width=True):
        st.session_state.quick_prompt = "Βρες μου 3 τελευταία νέα για την τεχνητή νοημοσύνη"
    
    if st.button("🎨 Εικόνα", use_container_width=True):
        st.session_state.quick_prompt = "Σχεδίασέ μου ένα ηλιοβασίλεμα σε ελληνικά νησιά"
    
    st.markdown("---")
    
    # Εργαλεία
    st.markdown("#### 🛠️ Εργαλεία")
    st.markdown("""
    - 🧮 Υπολογισμοί
    - 🕐 Ώρα & Ημερομηνία
    - 📝 Μέτρημα λέξεων
    - 🌐 Αναζήτηση web
    - 🤝 Research Agent
    - 💻 Εκτέλεση κώδικα
    - 🎨 Δημιουργία εικόνας
    """)
    
    st.markdown("---")
    
    # Κουμπιά
    st.markdown("#### 🎮 Ενέργειες")
    
    if st.button("🗑️ Καθάρισε τη μνήμη", use_container_width=True):
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        st.session_state.messages = []
        st.rerun()
    
    if mem:
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "📄 TXT",
                export_chat_txt(mem),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                use_container_width=True
            )
        with col_b:
            st.download_button(
                "📋 JSON",
                export_chat_json(mem),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                use_container_width=True
            )
    else:
        st.info("Δεν υπάρχει συνομιλία ακόμα.")
    
    st.markdown("---")
    st.caption("💡 Πες 'exit' για έξοδο")

# ===== ΣΥΝΟΜΙΛΙΑ =====

if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "messages" not in st.session_state:
    st.session_state.messages = load_memory()

# Μήνυμα καλωσορίσματος
if not st.session_state.messages:
    st.info("👋 Γεια σου! Γράψε κάτι για να ξεκινήσουμε. Θυμάμαι τα πάντα! 🧠")

# Δείξε όλα τα μηνύματα
for idx, msg in enumerate(st.session_state.messages):
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        
        # Κουμπί αντιγραφής
        with st.expander("📋 Αντιγραφή", expanded=False):
            st.code(msg["content"], language=None)
        
        # Smart Suggestions (μόνο στο τελευταίο assistant μήνυμα)
        if (msg["role"] == "assistant" 
            and idx == len(st.session_state.messages) - 1 
            and msg.get("suggestions")):
            
            st.markdown("💡 **Προτεινόμενες ερωτήσεις:**")
            sug_cols = st.columns(3)
            for i, sug in enumerate(msg["suggestions"][:3]):
                with sug_cols[i % 3]:
                    if st.button(f"👉 {sug}", key=f"sug_{idx}_{i}", use_container_width=True):
                        st.session_state.quick_prompt = sug
                        st.rerun()

# ===== ΕΙΣΑΓΩΓΗ =====

# 🎤 Voice Input
audio_input = st.audio_input("🎤 Ή μίλα αντί να γράψεις")

if audio_input:
    with st.spinner("🎧 Αναγνωρίζω τη φωνή..."):
        try:
            text = transcribe_audio(audio_input)
            if text:
                st.session_state.messages.append({"role": "user", "content": text})
                with st.chat_message("user", avatar="👤"):
                    st.write(text)

                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🤔 Σκέφτομαι..."):
                        config = {"configurable": {"thread_id": "my-session"}}
                        
                        try:
                            result = agent.invoke({"messages": st.session_state.messages}, config=config)
                        except Exception as e:
                            st.error(f"⚠️ DEBUG: {type(e).__name__}: {str(e)[:200]}")
                            st.stop()
                        
                        if result.get("__interrupt__"):
                            st.warning("⚠️ Ο agent θέλει να εκτελέσει κώδικα!")
                            action = result["__interrupt__"][0].value["action_requests"][0]
                            st.code(action["args"].get("command", ""), language="python")

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("✅ Έγκριση", key="voice_approve", use_container_width=True):
                                    result = agent.invoke(Command(resume={"type": "approve"}), config=config)
                                    st.rerun()
                            with col2:
                                if st.button("❌ Απόρριψη", key="voice_reject", use_container_width=True):
                                    result = agent.invoke(Command(resume={"type": "reject", "args": {"feedback": "Απορρίφθηκε"}}), config=config)
                                    st.rerun()
                        else:
                            answer = result["messages"][-1].content
                            st.write(answer)

                            if enable_tts:
                                try:
                                    audio_bytes = generate_tts_gtts(answer)
                                    st.audio(audio_bytes, format="audio/mp3")
                                except Exception as e:
                                    st.caption(f"⚠️ Αποτυχία TTS: {e}")

                st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    "suggestions": generate_suggestions(answer, st.session_state.messages)
})
                save_memory(st.session_state.messages)
        except Exception as e:
            st.error(f"⚠️ Αποτυχία αναγνώρισης: {e}")


# Έλεγχος για quick action
if "quick_prompt" in st.session_state and st.session_state.quick_prompt:
    prompt = st.session_state.quick_prompt
    st.session_state.quick_prompt = None
else:
    prompt = st.chat_input("Γράψε το μήνυμά σου...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Σκέφτομαι..."):
            config = {"configurable": {"thread_id": "my-session"}}
            try:
                result = agent.invoke({"messages": st.session_state.messages}, config=config)
            except Exception as e:
                st.error(f"⚠️ DEBUG: {type(e).__name__}: {str(e)[:200]}")
                st.stop()
            
            # Έλεγχος για interrupt (HITL)
            if result.get("__interrupt__"):
                st.warning("⚠️ Ο agent θέλει να εκτελέσει κώδικα!")
                action = result["__interrupt__"][0].value["action_requests"][0]
                st.code(action["args"].get("command", ""), language="python")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Έγκριση", key="main_approve", use_container_width=True):
                        result = agent.invoke(Command(resume={"type": "approve"}), config=config)
                        st.rerun()
                with col2:
                    if st.button("❌ Απόρριψη", key="main_reject", use_container_width=True):
                        result = agent.invoke(Command(resume={"type": "reject", "args": {"feedback": "Απορρίφθηκε"}}), config=config)
                        st.rerun()
                st.stop()
            else:
                answer = result["messages"][-1].content
                st.write(answer)
                
                # TTS
                if enable_tts:
                    try:
                        audio_bytes = generate_tts_gtts(answer)
                        st.audio(audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.caption(f"⚠️ Αποτυχία TTS: {e}")

    st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    "suggestions": generate_suggestions(answer, st.session_state.messages)
})
    save_memory(st.session_state.messages)