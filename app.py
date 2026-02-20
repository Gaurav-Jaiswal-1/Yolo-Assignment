import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Bottle Quality Detection", layout="centered")

st.title("🧴 Bottle Quality Detection System")
st.markdown("Detect whether a plastic bottle is **GOOD** or **BAD** using YOLO")

# ---------------- MODEL SELECTOR ----------------
model_option = st.selectbox(
    "Select Model Version",
    ("Non-Augmented Model","Augmented Model")
)

# ---------------- LOAD MODEL ----------------
try:
    if model_option == "Non-Augmented Model":
        model = YOLO(r"models\yolov8n\best.pt")
    else:
        model = YOLO(r"models\aug_model\best.pt")

    st.success("✅ Model loaded successfully")

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Show class names
st.write("Model Classes:", model.names)

# ---------------- CONFIDENCE SLIDER ----------------
conf_threshold = st.slider("Confidence Threshold", 0.01, 1.0, 0.15)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Bottle Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image)

    # ---------------- RUN INFERENCE ----------------
    with st.spinner("🔍 Analyzing bottle..."):
        results = model(img_array, conf=conf_threshold, imgsz=640)

    # Debug info
    st.write("🔎 Raw Detection Boxes:", results[0].boxes)

    # ---------------- CHECK DETECTIONS ----------------
    if results[0].boxes is not None and len(results[0].boxes) > 0:

        # Take highest confidence detection
        box = results[0].boxes[0]

        cls = int(box.cls)
        conf = float(box.conf)
        label = results[0].names[cls]

        annotated = results[0].plot(pil=True)


        # Show detection image
        st.image(annotated, caption="Detection Result", use_container_width=True)

        # ---------------- RESULT PANEL ----------------
        st.subheader("📊 Prediction Result")

        if label.lower() == "good":
            st.success("✅ Bottle Condition: GOOD")
        else:
            st.error("⚠️ Bottle Condition: BAD")

        st.write(f"Confidence Score: **{conf*100:.2f}%**")

        st.progress(conf)

    else:
        st.warning("❌ No bottle detected — try lowering confidence or using clearer image")

# ---------------- FOOTER ----------------
st.markdown("---")
