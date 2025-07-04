import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ────────────── CONFIG ──────────────
st.set_page_config(page_title="COVID-19 Chest Scan Classifier", layout="wide")

# ────────────── GLOBAL DARK THEME ──────────────
st.markdown("""
<style>
/* DARK BACKGROUND + BLUE ACCENTS */
html, body, [data-testid="stApp"] {
    background-color: #0F1A2B;
    color: #E8F1FF;
    font-family: 'Segoe UI', sans-serif;
}

/* HEADERS + LINKS */
h1, h2, h3, h4, h5, h6, a {
    color: #00BFFF !important;
}

/* TOP NAVBAR DARK */
[data-testid="stToolbar"] {
    background-color: #0F1A2B !important;
    color: #E8F1FF !important;
    border-bottom: 1px solid #1C2A40;
}

/* CARDS / BOXES */
.info-box, .result-box, .about-box {
    background-color: #1C2A40;
    padding: 1.2rem;
    border-radius: 12px;
    margin-top: 20px;
    box-shadow: 0 0 15px rgba(0,191,255,0.1);
}

/* PREDICTION BOX */
.result-label {
    font-size: 28px;
    font-weight: 700;
    color: #00BFFF;
}

/* CONFIDENCE BAR */
.confidence-bar {
    background-color: #112233;
    border-radius: 6px;
    height: 28px;
    overflow: hidden;
    margin-top: 12px;
}
.confidence-fill {
    height: 100%;
    text-align: center;
    line-height: 28px;
    font-weight: bold;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: #0F1A2B;
    transition: width 1s ease-in-out;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #101E33;
    color: #E8F1FF;
}
a {
    color: #4FC3F7;
}
</style>
""", unsafe_allow_html=True)

# ────────────── MODEL LOADING ──────────────
@st.cache_resource
def load_cnn_model():
    return load_model("covid19_cnn_model.h5")

model = load_cnn_model()

class_mapping = {
    0: "COVID-19 Positive",
    1: "Normal",
    2: "Viral Pneumonia"
}

# ────────────── IMAGE PREPROCESSING ──────────────
def preprocess_image(image, x_res=224, y_res=224):
    image = image.convert("RGB").resize((x_res, y_res))
    image_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image_array, axis=0)

# ────────────── PREDICTION ──────────────
def predict(image, model):
    preds = model.predict(image, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    return class_mapping[pred_idx], confidence, preds

# ────────────── UI TABS ──────────────
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Results", "ℹ️ About"])

# ────────────── HOME TAB ──────────────
with tab1:
    st.markdown("<h1 style='text-align:center;'>🩻 COVID-19 Chest Scan Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload a chest X-ray or CT scan to detect COVID-19, Pneumonia, or Normal findings.</p>", unsafe_allow_html=True)

    uploaded_img = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_img:
        image = Image.open(uploaded_img)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, use_column_width=True, caption="Uploaded Image")

        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("🖼️ Image Info")
            st.write(f"**Format**: {image.format}")
            st.write(f"**Mode**: {image.mode}")
            st.write(f"**Size**: {image.size[0]} x {image.size[1]}")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("🧠 Analyzing..."):
            img_array = preprocess_image(image)
            label, confidence, probs = predict(img_array, model)

        # ────────────── PREDICTION RESULT ──────────────
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">✅ Prediction: {label}</div>
            <p>This scan is predicted as <b>{label}</b> with a confidence of:</p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence:.1f}%;">{confidence:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state["probs"] = probs
    else:
        st.info("🧬 Please upload an image to get a prediction.")

# ────────────── RESULTS TAB ──────────────
with tab2:
    st.subheader("📊 Class Probabilities")

    if "probs" in st.session_state:
        labels = list(class_mapping.values())
        probabilities = st.session_state["probs"]

        fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#0F1A2B")
        bars = ax.barh(
            labels, probabilities,
            color=["#FF5252", "#4CAF50", "#00BCD4"],
            edgecolor="white"
        )
        ax.set_xlim(0, 1)
        ax.set_facecolor("#0F1A2B")
        ax.tick_params(colors="white")
        ax.set_xlabel("Probability", color="white")
        for bar, p in zip(bars, probabilities):
            ax.text(p + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{p*100:.1f}%", color="white", va="center", fontsize=10)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("📁 No predictions yet. Go to the Home tab to upload an image.")

# ────────────── ABOUT TAB ──────────────
with tab3:
    st.markdown("""
    <div class="about-box">
        <h3>ℹ️ About This App</h3>
        <p>This app uses a deep learning CNN model to classify chest X-rays into:</p>
        <ul>
            <li>🦠 COVID-19 Positive</li>
            <li>✅ Normal</li>
            <li>🤒 Viral Pneumonia</li>
        </ul>
        <h4>🧠 Model Info</h4>
        <ul>
            <li><b>Model:</b> CNN using Keras</li>
            <li><b>Input:</b> 224x224 RGB</li>
            <li><b>Framework:</b> TensorFlow</li>
        </ul>
        <h4>👨‍💻 Ai Engineer</h4>
        <p>
            Developed by <a href="https://github.com/Ziad-el3shry" target="_blank">Ziad Attia</a><br>
            Contact: ziadel3shry123@gmail.com
        </p>
        <p style="font-size: 13px; color: #AAA;">
        ⚠️ Disclaimer: For educational use only – not a certified diagnostic tool.
        </p>
    </div>
    """, unsafe_allow_html=True)
