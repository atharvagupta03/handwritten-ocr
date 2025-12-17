import streamlit as st
import google.generativeai as genai
from PIL import Image
import io


st.set_page_config(
    page_title="Handwritten OCR using Gemini Vision",
    layout="centered"
)

st.title("Handwritten Text Recognition")

# image upload
uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image_bytes = uploaded_file.read()
    handwritten_image = Image.open(io.BytesIO(image_bytes))

    st.image(handwritten_image, caption="Uploaded Image", use_column_width=True)

    #prompt
    ocr_prompt = """
    You are an OCR system specialized in handwritten documents.
    Carefully read the handwritten English text in the image.
    Ignore paper texture, background noise, and shadows.
    Return only the extracted text.
    Preserve line breaks wherever possible.
    Do not add explanations or summaries.
    """

    if st.button("🔍 Extract Text"):
        try:
            with st.spinner("Processing handwritten text..."):

                genai.configure(st.secrets["GEMINI_API_KEY"])

                model = genai.GenerativeModel("gemini-2.5-flash")

                response = model.generate_content(
                    [ocr_prompt, handwritten_image],
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 512
                    }
                )

            st.subheader(" Extracted Text")
            st.text_area(
                " Output",
                response.text,
                height=300
            )

        except Exception as e:
            st.error("error")
            st.exception(e)
