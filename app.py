import streamlit as st
import joblib

# Load trained model & vectorizer
model = joblib.load("flipkart_sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(
    page_title="Flipkart Sentiment Analysis",
    layout="centered"
)

# UI
st.title("🛒 Flipkart Review Sentiment Analysis")
st.write("Enter a product review to predict its sentiment.")

review = st.text_area(
    "Review Text",
    placeholder="Example: Very good quality and worth the price"
)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vec = tfidf.transform([review])
        prediction = model.predict(review_vec)

        if prediction[0] == 1:
            st.success("✅ Positive Review 😊")
        else:
            st.error("❌ Negative Review 😞")

st.markdown("---")
st.caption("Model: TF-IDF + Linear SVM")
