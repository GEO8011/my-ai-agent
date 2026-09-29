import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

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

# ===== ΜΟΝΤΕΛΟ =====

model = ChatOpenRouter(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ===== AGENT =====

agent = create_agent(
    model=model,
    tools=[calculate, get_time, word_count],
    system_prompt="Είσαι βοηθός. Χρησιμοποίησε τα εργαλεία όταν χρειάζεται."
)

# ===== ΣΥΝΟΜΙΛΙΑ =====

print("🤖 Ο agent είναι έτοιμος! Γράψε 'exit' για έξοδο.\n")

history = []

while True:
    user_input = input("Εσύ: ")

    if user_input.lower() == "exit":
        print("🤖 Αντίο!")
        break

    history.append({"role": "user", "content": user_input})

    response = agent.invoke({"messages": history})

    # Κρατάμε όλο το ιστορικό (και τις κλήσεις εργαλείων)
    history = response["messages"]

    print(f"🤖 Agent: {response['messages'][-1].content}\n")