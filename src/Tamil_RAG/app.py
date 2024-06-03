"""
MIT License

Author: Chandra S
Date: 2024-06-02

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
import os
import streamlit as st
from model import ChatModel
import rag_util
from encoder_wrapper import EmbeddingWrapper

FILES_DIR = "./LLM_RAG_Bot/files"

# Set up the page
st.set_page_config(page_title="Tamil Education - Student Learning RAG Chatbot",
                   page_icon=":school:",
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title("Tamil Education - Student Learning Platform RAG Chatbot")

# Available models
models = {
    "Microsoft Phi3-Tamilv0.5": "niranjanramarajar/Phi3-Tamil-v0-5",
    "LLaMA3 Tamilv05": "niranjanramarajar/Llama-3-Tamil-v0-5"
}

@st.cache_resource
def load_model(model_id, hf_token, device, quantization):
    model = ChatModel(model_id=model_id, hf_token=hf_token, device=device, quantization=quantization)
    return model

@st.cache_resource
def load_encoder():
    encoder = EmbeddingWrapper("BAAI/bge-m3")
    return encoder

def save_file(uploaded_file):
    """Helper function to save documents to disk."""
    if not os.path.exists(FILES_DIR):
        os.makedirs(FILES_DIR)
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Streamlit UI
st.sidebar.title('Model Selection')

# Device selection
device_choice = st.sidebar.selectbox("Select a device", ["CPU", "CUDA"], index=0)

# Quantization options based on device
if device_choice == "CPU":
    quantization_choice = st.sidebar.selectbox("Select quantization", ["fp32", "8bit", "4bit"], index=0)
else:
    quantization_choice = "fp16"

# Hugging Face Token (only if downloading the model)
download_model = st.sidebar.checkbox("Download model from Hugging Face", value=True)
hf_token = ""
if download_model:
    hf_token = st.sidebar.text_input("Enter your Hugging Face access token", type="password", key="hf_token_input")

# Model selection
model_choice = st.sidebar.selectbox("Select a model", list(models.keys()), key="model_select")

# Load or download the model
model_id = models[model_choice]

if st.sidebar.button('Load Model', key="load_model_button"):
    if not download_model or hf_token:
        try:
            st.session_state.model = load_model(model_id, hf_token, device_choice.lower(), quantization_choice)
            st.sidebar.success(f"Model {model_choice} loaded successfully on {device_choice} with {quantization_choice} quantization!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    else:
        st.sidebar.error("Please enter your Hugging Face access token.")

# Encoder
encoder = load_encoder()

# Document upload
uploaded_files = st.sidebar.file_uploader("Upload documents for model context", type=['pdf', 'docx', 'txt', 'csv', 'json'], accept_multiple_files=True, key="file_uploader")
file_paths = [save_file(file) for file in uploaded_files]

if 'db' not in st.session_state:
    st.session_state.db = None

if uploaded_files:
    new_docs = rag_util.load_and_split_documents(file_paths)
    if st.session_state.db is None:
        st.session_state.db = rag_util.FaissDb(docs=new_docs, embedding_function=encoder)
    else:
        new_docs_added = rag_util.add_new_documents_to_db(new_docs, st.session_state.db)
        if new_docs_added:
            st.sidebar.success(f"Added {len(new_docs_added)} new documents to the database.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('Ask me anything'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        user_prompt = st.session_state.messages[-1]['content']
        if st.session_state.db is not None:
            context = st.session_state.db.similarity_search(user_prompt, k=3)  # Default k value
        else:
            context = None
        
        if 'model' in st.session_state:
            concise_answer = "context" in prompt.lower() or "context-based" in prompt.lower()
            answer = st.session_state.model.inference(
                user_prompt, context=context, max_new_tokens=512, concise=concise_answer  # Default max_new_tokens value
            )

            response = st.write(answer)
        else:
            answer = "Model is not loaded. Please load a model first."
            response = st.write(answer)

    st.session_state.messages.append({'role': 'assistant', 'content': answer})