import streamlit as st
from transformers import pipeline

st.title('Приложение определяет эмоции в предложениях на русском языке.')

form = st.form(key='sentiment-form')
user_input = form.text_area('Введите Ваш текст')
submit = form.form_submit_button('Определить')

if submit:
    clf = pipeline(
        task = 'text-classification', 
        model = 'cointegrated/rubert-tiny2-cedr-emotion-detection')
     
    result = clf(user_input)


    for res in result:
       label = res['label']
       score = res['score']
       st.success(f'{label} sentiment (score: {score})')
 

        
