import streamlit as st
import requests
import json  # Added for parsing the data stream

# =============================================================================
# CONFIGURATION
# =============================================================================
# This URL will change every time you restart ngrok (on the free plan)
API_URL = "https://mandie-erasable-parallel.ngrok-free.dev"
MODEL = "mistral"


# =============================================================================
# API COMMUNICATION (NOW A STREAMING GENERATOR)
# =============================================================================

def send_message_stream(messages):
    """
    Sends the chat history to the API and yields the response tokens
    as they are generated.
    """
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 512,
                "stream": True  # <--- Key change: Tell the API to stream
            },
            stream=True  # <--- Key change: Tell `requests` to handle a stream
        )
        response.raise_for_status()  # Raise an exception for bad status codes

        # Handle the streaming response
        for line in response.iter_lines():
            if line:
                # Decode the line (it's in bytes)
                decoded_line = line.decode('utf-8')

                # Each chunk in the stream starts with "data: "
                if decoded_line.startswith('data: '):
                    # Strip the "data: " prefix
                    data_str = decoded_line[6:].strip()

                    # Check for the [DONE] signal
                    if data_str == '[DONE]':
                        break

                    try:
                        # Parse the JSON data chunk
                        data = json.loads(data_str)

                        # Find the content delta
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content")

                            # If content exists, yield it
                            if content:
                                yield content

                    except json.JSONDecodeError:
                        # Handle incomplete JSON chunks, just skip them
                        pass

    except requests.exceptions.RequestException as e:
        # Handle network errors (e.g., connection refused)
        st.error(f"⚠️ API Error: {e}")
        st.error("Is LM Studio running? Make sure the server is started and the API_URL is correct.")
        yield f"⚠️ API Error: {e}."  # Yield error to display it in the chat
    except Exception as e:
        yield f"⚠️ An unknown error occurred: {e}"


# =============================================================================
# CHAT RESET FUNCTION
# =============================================================================

def reset_chat():
    """
    Resets the chat history in Streamlit's session_state.
    It re-initializes the 'messages' list with the *current* system_prompt.
    """
    st.session_state.messages = [{"role": "system", "content": system_prompt}]


# =============================================================================
# STREAMLIT UI
# =============================================================================

# --- Page Setup ---
st.set_page_config(page_title="LM Studio Chat", page_icon="🤖", layout="centered")

st.title("🤖 LM Studio Local AI Chat")
st.write("Chat with your locally running model (e.g. Mistral) through LM Studio.")

# --- Persona Selection ---
personas = {
    "Default Assistant": "You are a helpful AI assistant.",
    "Friendly Tutor": "You are a patient and friendly teacher who explains clearly.",
    "Tech Expert": "You are a knowledgeable software engineer who answers concisely.",
    "Creative Writer": "You are an imaginative storyteller who writes vividly.",
    "Philosopher": "You are a thoughtful philosopher who gives deep insights.",
}

persona_choice = st.selectbox("🧠 Choose a Persona:", list(personas.keys()))
system_prompt = personas[persona_choice]

# --- Session State (Memory) Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

if st.session_state.messages[0]["content"] != system_prompt:
    st.session_state.messages[0]["content"] = system_prompt
    st.rerun()

# --- Display Chat History ---
for msg in st.session_state.messages[1:]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:  # "assistant"
        st.chat_message("assistant").write(msg["content"])

# --- Chat Input and Response Logic (Updated for Streaming) ---
if prompt := st.chat_input("Type your message here..."):

    # 1. Add user's message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display user's message in the chat UI
    st.chat_message("user").write(prompt)

    # 3. Get and display the assistant's streaming response
    with st.chat_message("assistant"):
        # `st.write_stream` takes a generator (our function) and
        # writes the yielded chunks to the UI as they arrive.
        # It also returns the *full, concatenated* string at the end.
        reply = st.write_stream(send_message_stream(st.session_state.messages))

    # 4. Add the *full* assistant response to the session state
    #    (Only if it wasn't an error)
    if not reply.startswith("⚠️"):
        st.session_state.messages.append({"role": "assistant", "content": reply})

# --- Reset Button ---
st.button("🧹 Reset Chat", on_click=reset_chat)