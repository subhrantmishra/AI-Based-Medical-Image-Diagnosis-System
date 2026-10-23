import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="AI Diagnosis System", layout="centered")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.big-title {
    font-size: 38px;
    font-weight: bold;
}
.result-card {
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🧠 AI Multi-Disease Diagnosis System</p>', unsafe_allow_html=True)

IMG_SIZE = 224

# ---------------- MODEL INFO ---------------- #
model_info = {
    "Pneumonia (X-ray)": ("medical_model.h5", "binary", "94%"),
    "Tuberculosis (X-ray)": ("tb_model.h5", "binary", "92%"),
    "Brain Tumor (MRI)": ("brain_tumor_model.h5", "multiclass", "96%"),
    "Lung Cancer (CT Scan)": ("ctscan_model.h5", "multiclass", "93%")
}

scan_type = st.selectbox("Select Scan Type", list(model_info.keys()))

model_path, class_type, accuracy = model_info[scan_type]

# ---------------- LOAD MODEL ---------------- #
with st.spinner("Loading AI Model..."):
    model = tf.keras.models.load_model(model_path)
    # Convert Sequential to Functional model (important for GradCAM)

st.success(f"Model Loaded ✅ | Accuracy: {accuracy}")
model = tf.keras.models.load_model(model_path, compile=False)

# FORCE BUILD (VERY IMPORTANT)
dummy = np.zeros((1,224,224,3))
_ = model.predict(dummy)

# ---------------- GRADCAM FUNCTION ---------------- #
def make_gradcam_heatmap(img_array, model):

    # Find last Conv layer index
    last_conv_index = None
    for i in range(len(model.layers)-1, -1, -1):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            last_conv_index = i
            break

    if last_conv_index is None:
        return None

    # Split model into two parts
    conv_model = tf.keras.Sequential(model.layers[:last_conv_index+1])
    classifier_model = tf.keras.Sequential(model.layers[last_conv_index+1:])

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)

        preds = classifier_model(conv_output)
        class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# ---------------- UPLOAD ---------------- #
uploaded_file = st.file_uploader("Upload Scan Image", type=["jpg","png","jpeg"])

if uploaded_file:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # -------- BINARY -------- #
    if class_type == "binary":
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            result = scan_type.split()[0].upper()
            color = "#ff4b4b"
            message = f"{result} DETECTED 😷 ({confidence*100:.2f}%)"
        else:
            result = "NORMAL"
            color = "#00c853"
            message = f"NORMAL ✅ ({(1-confidence)*100:.2f}%)"

    # -------- MULTICLASS -------- #
    else:
        if scan_type == "Brain Tumor (MRI)":
            classes = ["Glioma","Meningioma","No Tumor","Pituitary"]
        else:
            classes = ["Adenocarcinoma","Large Cell Carcinoma","Normal","Squamous Cell Carcinoma"]

        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        result = classes[class_index]

        if result.lower() in ["normal","no tumor"]:
            color = "#00c853"
            message = f"{result} ✅ ({confidence*100:.2f}%)"
        else:
            color = "#ff4b4b"
            message = f"{result} DETECTED 😷 ({confidence*100:.2f}%)"

    with col2:
        st.markdown(
            f'<div class="result-card" style="background-color:{color};">{message}</div>',
            unsafe_allow_html=True
        )

    # -------- GRADCAM -------- #
    heatmap = make_gradcam_heatmap(img_array, model)

    if heatmap is not None:
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(
            cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
            0.6,
            heatmap,
            0.4,
            0
        )

        st.subheader("🔍 Affected Area (GradCAM)")
        st.image(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # -------- PDF REPORT -------- #
    def generate_pdf():
        file_path = "report.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Medical AI Diagnosis Report", styles["Title"]))
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph(f"Scan Type: {scan_type}", styles["Normal"]))
        elements.append(Paragraph(f"Result: {result}", styles["Normal"]))
        elements.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles["Normal"]))

        doc.build(elements)
        return file_path

    pdf_path = generate_pdf()

    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download Medical Report 📄",
            data=f,
            file_name="Medical_Report.pdf",
            mime="application/pdf"
        )