import streamlit as st
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import google.generativeai as genai
import os

# -------------------------------
# 🔐 Setup Gemini API
# -------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Set via environment or manually below

# OPTIONAL: If you're not using environment variable, uncomment this:
genai.configure(api_key="AIzaSyAQqrBC0iCV9SXiSjTYEehsb6BAgDy7fIo")

model = genai.GenerativeModel("gemini-pro")

# -------------------------------
# 📷 OCR: Extract text from image
# -------------------------------
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_image(img: Image.Image) -> str:
    img_np = np.array(img)
    result = ocr.ocr(img_np, cls=True)
    extracted = ""
    for line in result[0]:
        extracted += line[1][0] + "\n"
    return extracted.strip()

# -------------------------------
# 🧠 Gemini Evaluation
# -------------------------------
def evaluate_with_gemini(extracted_text: str) -> dict:
    prompt = f"""
You are an AI evaluator. A student uploaded formulas for accuracy, precision, recall, and F1-score from a machine learning confusion matrix.

The formulas they submitted (extracted from an image) are:

{extracted_text}

Please evaluate each of the following:
- Accuracy formula (1 mark)
- Precision formula (1 mark)
- Recall formula (1 mark)
- F1-score formula (1 mark)
- Formatting and readability (1 mark)

Return the result in this JSON format:
{{
  "accuracy": 1 or 0,
  "precision": 1 or 0,
  "recall": 1 or 0,
  "f1_score": 1 or 0,
  "formatting": 1 or 0,
  "total": sum of above
}}
"""

    response = model.generate_content(prompt)
    try:
        import json
        result = json.loads(response.text)
        return result
    except Exception as e:
        return {"error": "Gemini returned unexpected format", "details": response.text}

# -------------------------------
# 🎛️ Streamlit UI
# -------------------------------
st.set_page_config(page_title="ML Formula Evaluator", layout="centered")
st.title("🧪 ML Confusion Matrix Formula Evaluator")

st.markdown("Upload an image of your formulas for **Accuracy**, **Precision**, **Recall**, and **F1-score**. This app will evaluate your formulas using OCR and Gemini AI.")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Extracting formulas using OCR..."):
        extracted_text = extract_text_from_image(img)

    st.subheader("📜 Extracted Text")
    st.code(extracted_text)

    with st.spinner("🧠 Evaluating using Gemini AI..."):
        result = evaluate_with_gemini(extracted_text)

    if "error" in result:
        st.error(result["error"])
        st.text(result["details"])
    else:
        st.success(f"✅ Total Score: {result['total']}/5")
        st.write("### Detailed Breakdown")
        for k in ["accuracy", "precision", "recall", "f1_score", "formatting"]:
            st.write(f"**{k.replace('_', ' ').capitalize()}**: {result[k]}/1")
