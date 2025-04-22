
import streamlit as st
import pandas as pd
from vertexai.generative_models import GenerativeModel
from backend_functions import get_answer_from_qa_system

# Load metadata files
text_metadata_df = pd.read_parquet("final_merge_text_data.parquet")
image_metadata_df = pd.read_parquet("final_merge_image_data.parquet")

st.set_page_config(page_title="Gemini QA Comparison", layout="wide")
st.title("🔍 MRAG System – Gemini Model Comparison (1.5 Flash / 1.5 Pro / 2.0 Flash)")

# Select models for side-by-side comparison
col1, col2 = st.columns(2)
with col1:
    model_choice_left = st.selectbox("Left Model", ["Gemini 1.5 Flash", "Gemini 2.0 Flash", "Gemini 1.5 Pro"], key="left")
with col2:
    model_choice_right = st.selectbox("Right Model", ["Gemini 1.5 Flash", "Gemini 2.0 Flash", "Gemini 1.5 Pro"], key="right")

def get_model_instance(choice):
    if choice == "Gemini 1.5 Flash":
        return GenerativeModel("gemini-1.5-flash-002")
    elif choice == "Gemini 2.0 Flash":
        return GenerativeModel("gemini-2.0-flash-001")
    else:
        return GenerativeModel("gemini-1.5-pro-002")

# Top-N sliders
top_n_text = st.slider("Top N Text Chunks", min_value=1, max_value=30, value=3)
top_n_image = st.slider("Top N Image Chunks", min_value=1, max_value=10, value=3)

# Input boxes for up to 3 questions
st.markdown("### Enter your questions for both models:")
q1 = st.text_area("📝 Question 1", height=80)
q2 = st.text_area("📝 Question 2", height=80)
q3 = st.text_area("📝 Question 3", height=80)

# Compare Answers
if st.button("🚀 Compare Answers"):
    query = "\n".join([q for q in [q1, q2, q3] if q.strip()])
    if not query:
        st.warning("Please enter at least one question.")
    else:
        with st.spinner("Generating responses from both models..."):
            try:
                model_left = get_model_instance(model_choice_left)
                model_right = get_model_instance(model_choice_right)

                response_left, *_ = get_answer_from_qa_system(
                    query=query,
                    text_metadata_df=text_metadata_df,
                    image_metadata_df=image_metadata_df,
                    top_n_text=top_n_text,
                    top_n_image=top_n_image,
                    model=model_left
                )
                response_right, *_ = get_answer_from_qa_system(
                    query=query,
                    text_metadata_df=text_metadata_df,
                    image_metadata_df=image_metadata_df,
                    top_n_text=top_n_text,
                    top_n_image=top_n_image,
                    model=model_right
                )

                left_col, right_col = st.columns(2)
                with left_col:
                    st.markdown(f"### 🧠 {model_choice_left}", unsafe_allow_html=True)
                    st.markdown(response_left, unsafe_allow_html=True)
                with right_col:
                    st.markdown(f"### 🧠 {model_choice_right}", unsafe_allow_html=True)
                    st.markdown(response_right, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")
