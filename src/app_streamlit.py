import streamlit as st
import os
import torch
import gdown
from inference import load_model, generate_reply

# --- Streamlit page setup ---
st.set_page_config(page_title="Urdu Chatbot", layout="centered")

st.title("🤖 اردو چیٹ بوٹ")
st.markdown(
    "<h4 style='direction:rtl;text-align:right;'>ٹرانسفارمر پر مبنی اردو چیٹ بوٹ</h4>",
    unsafe_allow_html=True
)

# --- Google Drive model setup ---
MODEL_PATH = "models/best_model.pt"
# your drive link converted to direct download
GDRIVE_URL = "https://drive.google.com/uc?id=12QWyxckQwuOMCgjAhobyH5t3PJmcuse9"

@st.cache_resource
def load_chatbot():
    # create models folder if not exists
    os.makedirs("models", exist_ok=True)
    
    # download model if not present
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🤖 ماڈل ڈاؤن لوڈ ہو رہا ہے، براہ کرم انتظار کریں..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    
    # now load using your existing function
    with st.spinner("🤖 ماڈل لوڈ ہو رہا ہے..."):
        model, vocab, inv_vocab, device = load_model()
    return model, vocab, inv_vocab, device

try:
    model, vocab, inv_vocab, device = load_chatbot()
    st.success("✅ ماڈل کامیابی سے لوڈ ہوگیا!")
except Exception as e:
    st.error(f"❌ ماڈل لوڈ کرنے میں مسئلہ ہوا: {e}")
    st.stop()

# --- Session state for chat history ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Urdu input box (RTL) ---
st.markdown("<br>", unsafe_allow_html=True)
user_input = st.text_input(
    "آپ کا سوال درج کریں:",
    key="input",
    placeholder="مثلاً آج موسم کیسا ہے؟",
)

# --- Send button action ---
if st.button("📨 بھیجیں"):
    if user_input.strip():
        with st.spinner("جواب تیار کیا جا رہا ہے..."):
            try:
                reply = generate_reply(user_input, model, vocab, inv_vocab, device)
                st.session_state.history.append(("👤 صارف:", user_input))
                st.session_state.history.append(("🤖 چیٹ بوٹ:", reply))
            except Exception as e:
                st.error(f"❌ جواب تیار کرنے میں مسئلہ ہوا: {e}")

# --- Display chat history (RTL) ---
for speaker, text in reversed(st.session_state.history):
    st.markdown(
        f"<p style='direction:rtl;text-align:right;'><b>{speaker}</b> {text}</p>",
        unsafe_allow_html=True
    )

# --- Footer ---
st.markdown(
    "<hr><p style='direction:rtl;text-align:center;'>پروجیکٹ 02 - اردو چیٹ بوٹ (NLP Assignment)</p>",
    unsafe_allow_html=True
)
