
import streamlit as st
import pandas as pd
from IPython.display import display
from vertexai.preview.generative_models import GenerativeModel
from backend_functions import get_answer_from_qa_system

# Load metadata files
text_metadata_df = pd.read_parquet("final_merge_text_data.parquet")
image_metadata_df = pd.read_parquet("final_merge_image_data.parquet")


st.set_page_config(page_title="Gemini QA System", layout="wide")
st.title("🔍 MRAG System (Gemini 1.5 Flash/Pro & 2.0 Flash)")

from vertexai.generative_models import GenerativeModel

# Select model
model_choice = st.selectbox("Select Gemini Model", ["Gemini 1.5 Flash", "Gemini 2.0 Flash", "Gemini 1.5 Pro"])
if model_choice == "Gemini 1.5 Flash":
    model_name = "gemini-1.5-flash-002"
elif model_choice == "Gemini 2.0 Flash":
    model_name = "gemini-2.0-flash-001"
else:
    model_name = "gemini-1.5-pro-002"
model = GenerativeModel(model_name)

# Top-N sliders
top_n_text = st.slider("Top N Text Chunks", min_value=1, max_value=30, value=1)
top_n_image = st.slider("Top N Image Chunks", min_value=1, max_value=10, value=1)

# Input fields
st.markdown("Enter up to three related questions to analyze:")
question_1 = st.text_area("📝 Question 1", height=80)
question_2 = st.text_area("📝 Question 2", height=80)
question_3 = st.text_area("📝 Question 3", height=80)

if st.button("🚀 Generate Answer"):
    query = "\n".join(q for q in [question_1, question_2, question_3] if q.strip())
    if not query:
        st.warning("Please enter at least one question.")
    else:
        with st.spinner("Analyzing and generating response..."):
            try:
                response, matched_text, matched_images = get_answer_from_qa_system(
                    query=query,
                    text_metadata_df=text_metadata_df,
                    image_metadata_df=image_metadata_df,
                    top_n_text=top_n_text,
                    top_n_image=top_n_image,
                    model=model
                )
                st.markdown("### 📖 Response", unsafe_allow_html=True)
                st.markdown(response, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
