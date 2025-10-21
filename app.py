import streamlit as st
import pickle
import sklearn

model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("📧 Email Spam/Ham Classifier")
st.write("Enter an email message below to check if it’s Spam or Ham (Not Spam).")

input_mail = st.text_area("✉️ Enter Email Text Here:")

if st.button("Predict"):
    if input_mail.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        input_data_features = vectorizer.transform([input_mail])
        prediction = model.predict(input_data_features)

        if prediction[0] == 1:
            st.success("✅ The email is **Ham (Not Spam)**")
        else:
            st.error("🚨 The email is **Spam**")

