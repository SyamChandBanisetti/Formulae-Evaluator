import streamlit as st
from PIL import Image
import pytesseract

from evaluator import evaluate_formulas  # your scoring logic

# -------------------------------
# OCR Function: Convert Image to Text
# -------------------------------
def extract_text_from_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="ML Formula Evaluator", layout="centered")
st.title("🧪 ML Confusion Matrix Formula Evaluator")

st.markdown("Upload an image of your formulas for Accuracy, Precision, Recall, and F1-score.")

# Upload image section
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Extracting formulas using OCR..."):
        text = extract_text_from_image(img)

    st.subheader("📜 Extracted Text from Image")
    st.text(text)

    with st.spinner("🧪 Evaluating formulas..."):
        scores, total = evaluate_formulas(text)

    st.success(f"✅ Total Score: {total}/5")

    st.write("### Detailed Breakdown:")
    for key, score in scores.items():
        st.write(f"**{key.capitalize()}**: {score}/1")
