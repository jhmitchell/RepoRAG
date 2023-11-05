import streamlit as st
import requests
import json

st.title('Repo Q&A')
st.write("Ask a question about the repo, and I'll do my best to answer it.")
question = st.text_input('Type your question here:')

if question:
    url = 'http://localhost:6000/reporag'
    payload = {"query": question}
    
    response = requests.post(url, json=payload)
    
    # Check for a valid response
    if response.status_code == 200:
        answer_data = response.json()
        answer = answer_data.get('Answer', 'No answer found')
        st.markdown(answer)
        reference = answer_data.get('Reference', 'No reference found')
        st.write('Reference:')
        st.code(reference)
    else:
        st.write(f'Error: {response.status_code}')
        st.write(f'Message: {response.text}')
