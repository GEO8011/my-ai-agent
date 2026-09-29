import os
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
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

load_dotenv()

MEMORY_FILE = "memory.json"

# ===== ΡΥΘΜΙΣΗ ΣΕΛΙΔΑΣ =====

st.set_page_config(
    page_title="Ο AI Agent μου",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    model=ChatOpenRouter(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        api_key=os.getenv("OPENROUTER_API_KEY")
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

# ===== ΜΟΝΤΕΛΟ & AGENT =====

@st.cache_resource
def get_agent():
    model = ChatOpenRouter(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    return create_agent(
        model=model,
        tools=[calculate, get_time, word_count, tavily, delegate_research, python_repl, generate_image],
        system_prompt="Είσαι ένας εξυπηρετικός βοηθός. Ο χρήστης είναι ο άνθρωπος που σου μιλάει — εσύ ΔΕΝ είσαι ο χρήστης. Χρησιμοποίησε τα εργαλεία όταν χρειάζεται.",
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
    st.markdown("### ⚡ AI Agent")
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
                        result = agent.invoke({"messages": st.session_state.messages}, config=config)

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

                st.session_state.messages.append({"role": "assistant", "content": answer})
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
            result = agent.invoke({"messages": st.session_state.messages}, config=config)
            
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

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_memory(st.session_state.messages)