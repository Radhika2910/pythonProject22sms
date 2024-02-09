import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from PIL import Image

# Load
image = Image.open("spam.png")

# Set the page configuration
page_bg_img = '''
<style>
body {
width:100;
background-image: url("spam.png");
background-size: 100%;

</style>
'''
page_bg_css = '''
<style>
body {
}
    margin: 150;
    padding: 500;
}
</style>
'''

st.markdown(page_bg_css, unsafe_allow_html=True)

# width manage
st.markdown(page_bg_img, unsafe_allow_html=True)

# Display
st.image(image, use_column_width=True)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Detection")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("ham")