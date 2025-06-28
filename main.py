import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from googletrans import Translator
import speech_recognition as sr

# ✅ Set Streamlit page config at the top
st.set_page_config(page_title="Smart City AI", layout="wide")

st.title("🌆 Smart City AI Sustainability Assistant")

# -------------------- ENV VARIABLES -------------------- #
PINECONE_AVAILABLE = False  # You disabled Pinecone usage.

# -------------------- LOCAL MODEL -------------------- #
@st.cache_resource
def load_local_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

llm = load_local_model()
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------- HELPERS -------------------- #
def query_llm(user_question):
    prompt = f"Answer the following question in detail:\n{user_question}"
    result = llm(prompt, max_new_tokens=512, do_sample=True, temperature=0.8, top_p=0.95)
    return result[0]['generated_text']

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def recognize_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now")
        try:
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError:
            st.error("Speech recognition failed.")
    return ""

def translate_text(text, dest_lang):
    translator = Translator()
    translated = translator.translate(text, dest=dest_lang)
    return translated.text

# -------------------- STREAMLIT UI -------------------- #
menu = st.sidebar.radio("Navigate", ["📄 Policy Assistant", "🧑 Citizen Tools", "📈 City Analytics"])

# -------------------- 1. POLICY ASSISTANT -------------------- #
if menu == "📄 Policy Assistant":
    st.header("📄 AI Policy Assistant")
    file = st.file_uploader("Upload Policy Document (PDF)", type="pdf")
    if file:
        text = extract_text_from_pdf(file)
        st.text_area("Extracted Text", text, height=200)

        if st.button("🔍 Summarize Policy"):
            summary = query_llm("Summarize this:\n" + text)
            st.success("Summary:")
            st.write(summary)

# -------------------- 2. CITIZEN TOOLS -------------------- #
elif menu == "🧑 Citizen Tools":
    st.header("🧑 Citizen Engagement")
    issue = st.text_area("Describe the issue")
    if st.button("📬 Submit Issue"):
        st.success("Thanks! Your report is noted.")

    if st.button("♻️ Generate Eco Tips"):
        tips = query_llm("Give 3 eco-friendly tips for city citizens.")
        st.info(tips)

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Ask a sustainability question")
    with col2:
        if st.button("🎤 Voice"):
            question = recognize_voice()

    lang = st.selectbox("🌐 Translate answer to", ["en", "hi", "te", "ta", "bn"])

    if st.button("💬 Ask") and question:
        answer = query_llm(question)
        translated = translate_text(answer, lang)
        st.success(translated)

# -------------------- 3. CITY ANALYTICS -------------------- #
elif menu == "📈 City Analytics":
    st.header("📊 City KPI Analysis")
    file = st.file_uploader("Upload KPI CSV (Date, Value)", type="csv")
    if file:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = (df['Date'] - df['Date'].min()).dt.days
        st.line_chart(df.set_index('Date')['Value'])

        model = LinearRegression().fit(df[['Time']], df['Value'])
        forecast = model.predict([[df['Time'].max() + 1]])[0]
        st.success(f"📈 Forecasted Next Value: {forecast:.2f}")

        iso = IsolationForest(contamination=0.1)
        df['Anomaly'] = iso.fit_predict(df[['Value']])
        st.subheader("🚨 Anomalies")
        st.dataframe(df[df['Anomaly'] == -1])

st.markdown("---")
st.caption("Built with 🤖 Transformers, 💡 FAISS-ready, and Streamlit")

