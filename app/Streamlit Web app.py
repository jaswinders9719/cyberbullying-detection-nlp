import os
import re
import numpy as np
import streamlit as st
import joblib
import shap
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# Optional BERT
BERT_AVAILABLE = False
try:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    BERT_AVAILABLE = True
except Exception:
    BERT_AVAILABLE = False

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Cyberbullying Classifier + SHAP + LIME", layout="wide")
st.title("Cyberbullying Tweet Classifier")
st.caption("Classical TF-IDF pipeline . Includes SHAP + LIME explanations.")

# ---------------------------
# NLTK setup (quiet but robust)
# ---------------------------
@st.cache_resource
def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")

ensure_nltk()
STOP_WORDS = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()

def clean_tweet_advanced(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#|[^\w\s]|\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "[EMPTY]"
    words = text.split()
    words = [LEMM.lemmatize(w) for w in words if w not in STOP_WORDS]
    out = " ".join(words).strip()
    return out if out else "[EMPTY]"

# ---------------------------
# Helpers: probability wrappers
# ---------------------------
def softmax(z):
    z = np.array(z)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)

def decision_to_proba(dec):
    dec = np.array(dec)
    # binary decision_function can be shape (n,) -> convert to (n,2)
    if dec.ndim == 1:
        dec = np.vstack([-dec, dec]).T
    return softmax(dec)

# ---------------------------
# Load classical artifacts
# ---------------------------
@st.cache_resource
def load_classical(best_model_path="best_model.pkl", encoder_path="label_encoder.pkl"):
    pipe = joblib.load(best_model_path)
    le = joblib.load(encoder_path)
    return pipe, le


def classical_predict_proba(pipe, texts):
    """
    Returns probs (n, K) even if classifier lacks predict_proba.
    """
    # Pipeline supports predict_proba only if final estimator does
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(texts)
        except Exception:
            pass

    # Fallback: decision_function -> softmax
    if hasattr(pipe, "decision_function"):
        dec = pipe.decision_function(texts)
        return decision_to_proba(dec)

    # Last resort: predict -> one-hot
    preds = pipe.predict(texts)
    n = len(texts)
    k = len(np.unique(preds))
    probs = np.zeros((n, k), dtype=float)
    for i, p in enumerate(preds):
        probs[i, int(p)] = 1.0
    return probs

def bert_predict_proba(tokenizer, model, device, texts, max_len=64):
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

# >>> TOXICITY ADDITION >>>
TOX_AVAILABLE = False

@st.cache_resource
def load_toxicity_artifacts(vec_path="toxicity_vectorizer.pkl", reg_path="rf_multi_reg.pkl"):
    vec = joblib.load(vec_path)
    reg = joblib.load(reg_path)
    return vec, reg

def toxicity_predict(reg_vec, reg_model, text):
    Xt = reg_vec.transform([text])
    y = reg_model.predict(Xt)  # (1,6) for your MultiOutputRegressor
    y = np.array(y).reshape(-1)
    return y

TOX_COLS = ["Insult", "Threat", "Identity_Attack", "Profanity", "Toxicity", "Severe_Toxicity"]
# <<< TOXICITY ADDITION <<<

# ---------------------------
# UI: model selection + load
# ---------------------------
with st.sidebar:
    st.header("Model")
    model_mode = st.radio(
        "Choose model",
        options=["Classical (best_model.pkl)"],
        index=0 if not (BERT_AVAILABLE and os.path.isdir("./distilbert_simple")) else 0
    )

    do_clean = st.checkbox("Apply cleaning (recommended)", value=True)
    show_debug = st.checkbox("Show debug info", value=False)

    # >>> TOXICITY ADDITION >>>
    st.divider()
    st.header("Toxicity Regression")
    enable_toxicity = st.checkbox("Show toxicity regression output", value=True)
    # <<< TOXICITY ADDITION <<<

# Load artifacts
pipe, le = load_classical()

bert_ok = BERT_AVAILABLE and os.path.isdir("./distilbert_simple")
if model_mode.startswith("DistilBERT") and not bert_ok:
    st.warning("DistilBERT not available. Install torch/transformers and ensure ./distilbert_simple exists. Falling back to Classical.")
    model_mode = "Classical (best_model.pkl)"

if model_mode.startswith("DistilBERT") and bert_ok:
    tokenizer, bert_model, bert_device = load_bert()

class_names = list(le.classes_)
lime_explainer = LimeTextExplainer(class_names=class_names)

# >>> TOXICITY ADDITION >>>
tox_vec = None
tox_reg = None
tox_ok = False
if enable_toxicity:
    try:
        tox_ok = os.path.exists("toxicity_vectorizer.pkl") and os.path.exists("rf_multi_reg.pkl")
        if tox_ok:
            tox_vec, tox_reg = load_toxicity_artifacts("toxicity_vectorizer.pkl", "rf_multi_reg.pkl")
        else:
            st.sidebar.warning("Toxicity files not found: toxicity_vectorizer.pkl and/or rf_multi_reg.pkl")
    except Exception as e:
        st.sidebar.error(f"Failed to load toxicity regressor: {repr(e)}")
        tox_ok = False
# <<< TOXICITY ADDITION <<<

# ---------------------------
# Input
# ---------------------------
default_text = "You are such a loser. Nobody likes you."
raw_text = st.text_area("Enter tweet text", value=default_text, height=140)

text_used = clean_tweet_advanced(raw_text) if do_clean else (raw_text.strip() if raw_text.strip() else "[EMPTY]")

colA, colB = st.columns([1, 1], gap="large")

# ---------------------------
# Predict
# ---------------------------
if st.button("Predict + Explain", type="primary"):

    # --- classification prediction ---
    if model_mode.startswith("Classical"):
        probs = classical_predict_proba(pipe, [text_used])[0]
        pred_id = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_id])[0]
    else:
        probs = bert_predict_proba(tokenizer, bert_model, bert_device, [text_used])[0]
        pred_id = int(np.argmax(probs))
        pred_label = class_names[pred_id]

    with colA:
        st.subheader("Prediction")
        st.write(f"**Text used:** {text_used}")
        st.write(f"**Predicted class:** {pred_label}")
        st.write("**Probabilities:**")
        for i, cn in enumerate(class_names):
            st.write(f"{cn}: {probs[i]:.4f}")

        # >>> TOXICITY ADDITION >>>
        if enable_toxicity:
            st.divider()
            st.subheader("Toxicity Regression (No SHAP/LIME)")
            if tox_ok:
                # Important: toxicity model was trained on its own cleaning function in your notebook.
                # If you used different cleaning there, align it here.
                tox_input = text_used  # using same cleaned text for consistency
                tox_vals = toxicity_predict(tox_vec, tox_reg, tox_input)

                for name, val in zip(TOX_COLS, tox_vals):
                    st.write(f"**{name}:** {float(val):.4f}")
            else:
                st.info("Toxicity regression not available (missing pkl files).")
        # <<< TOXICITY ADDITION <<<

    # ---------------------------
    # LIME (classification only)
    # ---------------------------
    with colB:
        st.subheader("LIME Explanation (Classification)")
        try:
            if model_mode.startswith("Classical"):
                def lime_fn(x):
                    x = [clean_tweet_advanced(t) if do_clean else (t.strip() if t.strip() else "[EMPTY]") for t in x]
                    return classical_predict_proba(pipe, x)
            else:
                def lime_fn(x):
                    x = [clean_tweet_advanced(t) if do_clean else (t.strip() if t.strip() else "[EMPTY]") for t in x]
                    return bert_predict_proba(tokenizer, bert_model, bert_device, x)

            lime_exp = lime_explainer.explain_instance(
                text_used,
                classifier_fn=lime_fn,
                num_features=12
            )
            lime_html = lime_exp.as_html()
            components.html(lime_html, height=420, scrolling=True)
        except Exception as e:
            st.error(f"LIME failed: {repr(e)}")

    st.divider()

    # ---------------------------
    # SHAP (classification only)
    # ---------------------------
    st.subheader("SHAP Explanation (token impact) — Classification Only")

    try:
        masker = shap.maskers.Text()

        if model_mode.startswith("Classical"):
            def shap_fn(texts):
                texts = [clean_tweet_advanced(t) if do_clean else (t.strip() if t.strip() else "[EMPTY]") for t in texts]
                return classical_predict_proba(pipe, texts)
            explainer = shap.Explainer(shap_fn, masker, output_names=class_names)
        else:
            def shap_fn(texts):
                texts = [clean_tweet_advanced(t) if do_clean else (t.strip() if t.strip() else "[EMPTY]") for t in texts]
                return bert_predict_proba(tokenizer, bert_model, bert_device, texts)
            explainer = shap.Explainer(shap_fn, masker, output_names=class_names)

        sv = explainer([text_used])

        out = shap.plots.text(sv[0], display=False)
        if hasattr(out, "data"):
            html = out.data
        elif isinstance(out, str):
            html = out
        else:
            html = str(out)

        components.html(shap.getjs() + html, height=420, scrolling=True)

    except Exception as e:
        st.error(f"SHAP failed: {repr(e)}")
        st.info("If SHAP fails on your machine, update shap: `pip install -U shap`")

    if show_debug:
        st.divider()
        st.subheader("Debug")
        st.write("Model mode:", model_mode)
        st.write("Cleaning:", do_clean)
        # >>> TOXICITY ADDITION >>>
        st.write("Toxicity enabled:", enable_toxicity)
        st.write("Toxicity artifacts loaded:", tox_ok)
        # <<< TOXICITY ADDITION <<<
