import os
# Fix DeBERTa-v3 spm.model protobuf parsing error — must be set before any imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from src.inference import predict_single

# Path to local weights folder
LOCAL_WEIGHTS_DIR = "weights"

st.set_page_config(page_title="MCQ Solver", page_icon="🧠", layout="centered")

st.title("🧠 Multiple Choice Question (MCQ) Solver")
st.markdown("Fast, accurate MCQ answering engine using lightweight scratch BiLSTM-Attention and pre-trained Transformer models.")

with st.form("mcq_form"):
    question = st.text_area("Enter your question here", placeholder="e.g. Which planet is known as the Red Planet?")
    
    st.markdown("**Choices**")
    r1_1, r1_2 = st.columns(2)
    choice_a = r1_1.text_input("Choice A", placeholder="Earth")
    choice_b = r1_2.text_input("Choice B", placeholder="Mars")
    
    r2_1, r2_2 = st.columns(2)
    choice_c = r2_1.text_input("Choice C", placeholder="Jupiter")
    choice_d = r2_2.text_input("Choice D", placeholder="Venus")
    
    choice_e = st.text_input("Choice E", placeholder="Saturn")
    
    choices = [choice_a, choice_b, choice_c, choice_d, choice_e]
    
    # Dropdown to choose architecture — Scratch as default
    arch = st.selectbox(
        "Select Model Architecture", 
        ["scratch", "deberta", "roberta"],
        index=0,
        help="Scratch is a lightweight custom BiLSTM model. DeBERTa and RoBERTa are powerful Transformer models with deep semantic knowledge."
    )
    
    submit = st.form_submit_button("Solve MCQ")

if submit:
    if not question.strip():
        st.warning("Please enter a question.")
    elif any(not c.strip() for c in choices):
        st.warning("Please fill in all 5 choices (A-E).")
    else:
        with st.spinner(f"Running inference with {arch.upper()} model..."):
            try:
                predictions = predict_single(
                    question=question, 
                    choices=choices, 
                    weights_dir=LOCAL_WEIGHTS_DIR, 
                    architecture=arch
                )
                
                st.success("Inference completed!")
                st.subheader("📊 Ranked Predictions")
                
                # Highlight top predicted choice
                top_label, top_conf = predictions[0]
                top_choice_text = choices[ord(top_label) - ord('A')]
                st.info(f"**Recommended Answer: Choice {top_label}** ({top_choice_text}) — Confidence: **{top_conf:.2%}**")
                
                # Note if scratch model was used for general knowledge questions
                if arch == "scratch":
                    st.caption("💡 *Tip: For general knowledge questions, try selecting **deberta** or **roberta** in the dropdown for deeper semantic understanding!*")
                
                # List all choices with progress bars
                for label, confidence in predictions:
                    idx = ord(label) - ord('A')
                    st.write(f"**Choice {label}** ({choices[idx]}) — Confidence: `{confidence:.2%}`")
                    st.progress(float(confidence))
            except Exception as e:
                st.error(f"Inference error: {e}")