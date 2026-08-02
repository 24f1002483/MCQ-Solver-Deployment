import streamlit as st
import os
from huggingface_hub import hf_hub_download
from src.inference import predict_single

# Configuration
REPO_ID = "NidhiHe/MCQ-solver"
LOCAL_WEIGHTS_DIR = "weights"

@st.cache_resource
def download_weights():
    os.makedirs(LOCAL_WEIGHTS_DIR, exist_ok=True)
    architectures = ["deberta", "roberta", "scratch"]
    folds = range(5)
    
    # Use st.status to show progress in the UI
    with st.status("Initializing model weights...", expanded=True) as status:
        for arch in architectures:
            for fold in folds:
                filename = f"{arch}_best_fold_{fold}.pt"
                # Update the message dynamically
                status.write(f"Checking {filename}...")
                
                hf_hub_download(
                    repo_id=REPO_ID, 
                    filename=f"weights/{filename}", 
                    local_dir="."
                )
        status.update(label="All weights ready!", state="complete", expanded=False)
    
    return LOCAL_WEIGHTS_DIR

# Initialize weights (this downloads them if they don't exist)
weights_path = download_weights()

st.title("MCQ Solver")

with st.form("mcq_form"):
    question = st.text_area("Question")
    c1, c2 = st.columns(2)
    choices = [
        c1.text_input("Choice A"), c2.text_input("Choice B"), 
        c1.text_input("Choice C"), c2.text_input("Choice D"), 
        st.text_input("Choice E")
    ]
    arch = st.selectbox("Select Model Architecture", ["deberta", "roberta", "scratch"])
    submit = st.form_submit_button("Solve")

if submit:
    # Use predict_single as requested
    with st.spinner('Running ensemble inference...'):
        try:
            predictions = predict_single(
                question=question, 
                choices=choices, 
                weights_dir=weights_path, 
                architecture=arch
            )
            
            st.subheader("Ranked Predictions")
            for i, (label, confidence) in enumerate(predictions):
                st.write(f"**#{i+1} {label}** — Confidence: {confidence:.2%}")
        except Exception as e:
            st.error(f"An error occurred during inference: {e}")
