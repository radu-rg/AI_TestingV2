import json  # Added for parsing the data stream
import os
from urllib.parse import urlparse

import requests
import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================
# This URL will change every time you restart ngrok (on the free plan)
DEFAULT_API_URL = os.environ.get("LMSTUDIO_API_URL", "https://mandie-erasable-parallel.ngrok-free.dev")
DEFAULT_MODEL = os.environ.get("LMSTUDIO_MODEL", "mistral")

PREDEFINED_MODELS = [
    "mistral",
    "llama-3",
    "phi-2",
]


# =============================================================================
# API COMMUNICATION (NOW A STREAMING GENERATOR)
# =============================================================================

def send_message_stream(messages, api_url, model):
    """
    Sends the chat history to the API and yields the response tokens
    as they are generated.
    """
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
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
        st.error(
            f"Is LM Studio running? Make sure the server at {api_url} is started and the API URL is correct."
        )
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

# --- Connection Defaults & Sidebar Configuration ---
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL

url_is_valid = False
custom_model_is_valid = True
chosen_model = st.session_state.model

with st.sidebar:
    st.header("Connection Settings")

    api_url_input = st.text_input("LM Studio API URL", value=st.session_state.api_url)
    stripped_api_url = api_url_input.strip()

    if not stripped_api_url:
        st.error("API URL is required.")
    else:
        parsed_url = urlparse(stripped_api_url)
        if parsed_url.scheme and parsed_url.netloc:
            url_is_valid = True
        else:
            st.error("Please enter a valid API URL (including scheme, e.g., https://host).")

    model_options = list(PREDEFINED_MODELS)
    if st.session_state.model not in model_options:
        model_options.insert(0, st.session_state.model)
    model_options.append("Custom")

    model_choice = st.selectbox(
        "Model",
        options=model_options,
        index=model_options.index(st.session_state.model)
        if st.session_state.model in model_options
        else len(model_options) - 1,
    )

    if model_choice == "Custom":
        custom_model_name = st.text_input(
            "Custom model name",
            value="" if st.session_state.model in PREDEFINED_MODELS else st.session_state.model,
        ).strip()

        if custom_model_name:
            chosen_model = custom_model_name
        else:
            st.error("Custom model name cannot be empty.")
            custom_model_is_valid = False
    else:
        chosen_model = model_choice

has_validation_errors = not url_is_valid or not custom_model_is_valid

if url_is_valid:
    st.session_state.api_url = stripped_api_url

if custom_model_is_valid:
    st.session_state.model = chosen_model

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

    if has_validation_errors:
        with st.chat_message("assistant"):
            st.error("Please resolve the sidebar configuration errors before sending a message.")
    else:
        # 1. Add user's message to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Display user's message in the chat UI
        st.chat_message("user").write(prompt)

        # 3. Get and display the assistant's streaming response
        with st.chat_message("assistant"):
            # `st.write_stream` takes a generator (our function) and
            # writes the yielded chunks to the UI as they arrive.
            # It also returns the *full, concatenated* string at the end.
            reply = st.write_stream(
                send_message_stream(
                    st.session_state.messages,
                    st.session_state.api_url,
                    st.session_state.model,
                )
            )

        # 4. Add the *full* assistant response to the session state
        #    (Only if it wasn't an error)
        if not reply.startswith("⚠️"):
            st.session_state.messages.append({"role": "assistant", "content": reply})

# --- Reset Button ---
st.button("🧹 Reset Chat", on_click=reset_chat)

