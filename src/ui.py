import streamlit as st
from app import RAGAssistant, load_documents

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Assistant")
st.caption("Ask questions based on your document knowledge base")

# -------------------------
# Initialize assistant once
# -------------------------
@st.cache_resource
def init_assistant():
    assistant = RAGAssistant()
    docs = load_documents()
    assistant.add_documents(docs)
    return assistant

assistant = init_assistant()

# -------------------------
# SESSION STATE (CHAT HISTORY)
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

# -------------------------
# Display chat history
# -------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# -------------------------
# User input (ChatGPT-style)
# -------------------------
prompt = st.chat_input("Ask a question...")

if prompt:
    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = assistant.invoke(prompt)
            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
