import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("C:/Users/rammohan\PycharmProjects\pythonProject1\Data_Science\Email-spam classifier\spam.csv", encoding="ISO-8859-1")
    data2 = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1)
    data2.rename(columns={"v1": "Spam or Ham", "v2": "Message"}, inplace=True)
    return data2

data = load_data()

# Preprocess the data
def preprocess_data(data):
    # Convert the categorical values into numerical values
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    data['Spam or Ham'] = encoder.fit_transform(data['Spam or Ham'])
    # Remove duplicates
    data = data.drop_duplicates(keep='first')
    return data

data = preprocess_data(data)

# Split the data
X = data['Message']
y = data['Spam or Ham']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_count, y_train)

# Create the Streamlit application
st.title("Email Spam Classification")
st.subheader("Enter the email message to classify:")

mail_input = st.text_input("Email Message:", "", help="Enter the email message to classify")

if st.button('Classify'):
    mail_input = [mail_input]
    mail_input_count = cv.transform(mail_input)
    prediction = model.predict(mail_input_count)
    if prediction[0] == 0:
        st.write("The email message is classified as: Ham")
    else:
        st.write("The email message is classified as: Spam")

