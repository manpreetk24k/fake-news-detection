import streamlit as st
import pickle
import os

# Load pickle model
def load_pickle_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)

# Load models and vectorizer
model_path = "models"
LR = load_pickle_model(os.path.join(model_path, "lr_model.pkl"))
DT = load_pickle_model(os.path.join(model_path, "dt_model.pkl"))
GBC = load_pickle_model(os.path.join(model_path, "gbc_model.pkl"))
RFC = load_pickle_model(os.path.join(model_path, "rfc_model.pkl"))
vectorizer = load_pickle_model(os.path.join(model_path, "vectorizer.pkl"))

# Word preprocessing (can be extended)
def wordopt(text):
    return text.lower()

# Convert output label
def output_label(n):
    return "Fake News" if n == 0 else "Real News"

# Prediction function
def manual_testing(news):
    news = wordopt(news)
    vect_news = vectorizer.transform([news])
    pred_LR = LR.predict(vect_news)[0]
    pred_DT = DT.predict(vect_news)[0]
    pred_GBC = GBC.predict(vect_news)[0]
    pred_RFC = RFC.predict(vect_news)[0]

    return {
        "Logistic Regression": output_label(pred_LR),
        "Decision Tree": output_label(pred_DT),
        "Gradient Boosting": output_label(pred_GBC),
        "Random Forest": output_label(pred_RFC)
    }

# Streamlit UI
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter news text below and see predictions from 4 machine learning models.")

news_input = st.text_area("Enter News Text", height=200)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            results = manual_testing(news_input)
            st.success("Prediction Complete!")
            for model, result in results.items():
                st.write(f"**{model}** predicts: `{result}`")
