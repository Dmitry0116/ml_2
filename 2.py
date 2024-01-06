import streamlit as st
from transformers import pipeline

st.title('Приложение определяет настроение в предложениях на русском языке.')

form = st.form(key='sentiment-form')
user_input = form.text_area('Введите Ваш текст')
submit = form.form_submit_button('Определить')

if submit:
    classifier = pipeline("sentiment-analysis")
    result = classifier(user_input)[0]
    label = result['label']
    score = result['score']

    if label == 'POSITIVE':
        st.success(f'{label} sentiment (score: {score})')
    else:
        st.error(f'{label} sentiment (score: {score})')
        
