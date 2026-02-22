import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Bottle Quality Classification", layout="centered")

st.title("🧴 Bottle Quality Detection System")
st.markdown("Classify whether a plastic bottle is **GOOD** or **BAD**")

# ---------------- LOAD MODEL ----------------
try:
    model = YOLO("models/classifier/best.pt")  # 🔁 change path if needed
    st.success("✅ Model loaded successfully")

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Show class names
st.write("Model Classes:", model.names)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Bottle Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    # ---------------- RUN INFERENCE ----------------
    with st.spinner("🔍 Analyzing bottle..."):
        results = model(img_array)

    probs = results[0].probs

    if probs is not None:

        # Get prediction
        top1 = probs.top1
        confidence = float(probs.top1conf)
        label = results[0].names[top1]

        # ---------------- RESULT PANEL ----------------
        st.subheader("📊 Prediction Result")

        if label.lower() == "good":
            st.success("✅ Bottle Condition: GOOD")
        else:
            st.error("⚠️ Bottle Condition: BAD")

        st.write(f"Confidence Score: **{confidence*100:.2f}%**")
        st.progress(confidence)

        # Show probability distribution
        st.subheader("📈 Class Probabilities")

        for i, prob in enumerate(probs.data):
            class_name = results[0].names[i]
            st.write(f"{class_name}: {float(prob)*100:.2f}%")

    else:
        st.warning("❌ Unable to classify image")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using YOLOv8 Classification")