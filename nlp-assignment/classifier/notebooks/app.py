import streamlit as st
import os
import joblib
import nltk

# --- CONFIGURATION ---
PAGE_TITLE = "Fake News Detector"
PAGE_ICON = "🕵️‍♀️"
MODEL_PATH = os.path.join('models', 'best_text_model.joblib')
NLTK_DIR = os.path.join(os.getcwd(), 'nltk_data')

# --- 1. SETUP NLTK (MUST BE DONE BEFORE IMPORTING PREPROCESSOR) ---
# We configure NLTK first because preprocessor.py loads stopwords immediately.
def setup_nltk():
    # 1. Add the local nltk_data folder to NLTK's search path
    if os.path.exists(NLTK_DIR):
        nltk.data.path.append(NLTK_DIR)
    
    # 2. Ensure resources exist (Download if missing)
    resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            # Check if resource is available
            nltk.data.find(f'corpora/{res}' if res in ['stopwords', 'wordnet', 'omw-1.4'] else f'tokenizers/{res}')
        except LookupError:
            # If not found, download it to the local directory
            if not os.path.exists(NLTK_DIR):
                os.makedirs(NLTK_DIR, exist_ok=True)
            nltk.download(res, download_dir=NLTK_DIR, quiet=True)

setup_nltk()

# Ensure the required NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- 2. IMPORT CUSTOM MODULES ---
# Now that NLTK is ready, we can safely import your preprocessor
try:
    import preprocessor
except ImportError as e:
    st.error(f"❌ Critical Error: Could not import 'preprocessor.py'. Ensure it is in the same directory as app.py.\n\nDetails: {e}")
    st.stop()

# --- 3. STREAMLIT APP LOGIC ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        # joblib will automatically find the TextPreprocessor class via the import above
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("""
    **Assignment 2.1: Model Deployment** Paste a news article snippet below to classify it as **REAL** or **FAKE**.
    """)
    
    # Load Model
    model = load_model()
    
    if model is None:
        st.error(f"⚠️ Model file not found at: `{MODEL_PATH}`. Please ensure you have run the notebook to generate the model.")
        return

    # Input Area
    text_input = st.text_area("Enter News Text:", height=200, placeholder="Type or paste the news article content here...")

    # Prediction Button
    if st.button("Predict Authenticity", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text patterns..."):
                try:
                    # 1. Predict
                    prediction = model.predict([text_input])[0]
                    
                    # 2. Get Confidence (if available)
                    confidence = 0.0
                    if hasattr(model, "predict_proba"):
                        confidence = model.predict_proba([text_input]).max()

                    # 3. Display Result
                    st.divider()
                    
                    # Mapping: Assuming 0=Fake, 1=Real (Standard for these datasets)
                    # If your model outputs text strings directly, this logic adapts automatically.
                    is_real = (prediction == 1 or prediction == "REAL" or prediction == "True")
                    
                    if is_real:
                        st.success("## ✅ Prediction: REAL NEWS")
                        if confidence > 0:
                            st.caption(f"Confidence Score: {confidence:.2%}")
                    else:
                        st.error("## 🚨 Prediction: FAKE NEWS")
                        if confidence > 0:
                            st.caption(f"Confidence Score: {confidence:.2%}")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()