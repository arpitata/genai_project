import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Title
# -------------------------------
st.title("Gen AI App – Transformers Demo")

st.write("This app demonstrates text generation, summarization, sentiment analysis, NER, QA, translation, paraphrasing, grammar correction, and text similarity.")

# -------------------------------
# Load Models Once
# -------------------------------
@st.cache_resource
def load_models():
    models = {
        "generator": pipeline("text-generation", model="gpt2"),
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn"),
        "sentiment": pipeline("sentiment-analysis"),
        "ner": pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple"),
        "qa": pipeline("question-answering"),
        "translate": pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
        "paraphrase": pipeline("text2text-generation", model="t5-small"),
        "grammar": pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1"),
        "sim_model": SentenceTransformer("all-MiniLM-L6-v2")
    }
    return models

models = load_models()

# -------------------------------
# Sidebar Menu
# -------------------------------
menu = st.sidebar.selectbox(
    "Select a Task",
    [
        "Text Generation",
        "Summarization",
        "Sentiment Analysis",
        "Named Entity Recognition (NER)",
        "Question Answering",
        "Translation (EN → FR)",
        "Paraphrasing",
        "Grammar Correction",
        "Text Similarity"
    ]
)

# -------------------------------
# Text Generation
# -------------------------------
if menu == "Text Generation":
    text = st.text_input("Enter a prompt:")
    if st.button("Generate"):
        output = models["generator"](text, max_length=200)
        st.write(output[0]["generated_text"])

# -------------------------------
# Summarization
# -------------------------------
elif menu == "Summarization":
    text = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        result = models["summarizer"](text, max_length=50, min_length=10, do_sample=False)
        st.success(result[0]["summary_text"])

# -------------------------------
# Sentiment
# -------------------------------
elif menu == "Sentiment Analysis":
    text = st.text_input("Enter a sentence:")
    if st.button("Analyze"):
        result = models["sentiment"](text)
        st.write(result)

# -------------------------------
# NER
# -------------------------------
elif menu == "Named Entity Recognition (NER)":
    text = st.text_input("Enter text:")
    if st.button("Extract Entities"):
        result = models["ner"](text)
        st.json(result)

# -------------------------------
# Question Answering
# -------------------------------
elif menu == "Question Answering":
    question = st.text_input("Enter your question:")
    context = st.text_area("Enter context:")
    if st.button("Get Answer"):
        result = models["qa"](question=question, context=context)
        st.write(result["answer"])

# -------------------------------
# Translation
# -------------------------------
elif menu == "Translation (EN → FR)":
    text = st.text_input("Enter English text:")
    if st.button("Translate"):
        result = models["translate"](text)
        st.success(result[0]["translation_text"])

# -------------------------------
# Paraphrasing
# -------------------------------
elif menu == "Paraphrasing":
    text = st.text_area("Enter text to paraphrase:")
    if st.button("Paraphrase"):
        out = models["paraphrase"](f"paraphrase: {text}")
        st.success(out[0]["generated_text"])

# -------------------------------
# Grammar Correction
# -------------------------------
elif menu == "Grammar Correction":
    text = st.text_input("Enter incorrect sentence:")
    if st.button("Correct Grammar"):
        result = models["grammar"](text)
        st.success(result[0]["generated_text"])

# -------------------------------
# Text Similarity
# -------------------------------
elif menu == "Text Similarity":
    a = st.text_input("Sentence 1:")
    b = st.text_input("Sentence 2:")
    if st.button("Check Similarity"):
        vec1 = models["sim_model"].encode(a, convert_to_tensor=True)
        vec2 = models["sim_model"].encode(b, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(vec1, vec2)
        st.write("Similarity Score:", float(sim))
