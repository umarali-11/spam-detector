import streamlit as st
import joblib

# Load the saved model
model = joblib.load("model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("vectorizer.pkl")


# Website title
st.title("📧 Email Spam Detector")

# Description
st.write("Enter an SMS or email message below to check whether it is Spam or Not Spam.")


# Text box
message = st.text_area("Enter your message:")


# Button
if st.button("Check Message"):

    # Check if the user entered something
    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        # Convert message into TF-IDF numbers
        message_tfidf = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_tfidf)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 Spam")

            st.write(
                "The model predicts that this message is likely Spam."
            )

        else:
            st.success("✅ Not Spam")

            st.write(
                "The model predicts that this message is likely Not Spam."
            )
            