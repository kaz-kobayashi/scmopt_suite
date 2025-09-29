from langchain.llms import OpenAI
import streamlit as st
from langchain_openai import ChatOpenAI 

import sys 
import pathlib
sys.path.append('..')
p = pathlib.Path(".") #現在のフォルダ
scriptfolder = p / "scripts"

from scmopt2.llm import *

st.title('🦜🔗 SCM AI')

client = ChatOpenAI(temperature=0.0, base_url="http://localhost:1234/v1", api_key="not-needed", model="local-model")

#st.write(opening_message)
st.write(vrp_message)

def generate_response(input_text):
    llm = client
    st.info(llm.invoke(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:')
    submitted = st.form_submit_button('Submit')
    if submitted:
        #generate_response(text)
        #ret = extract_sc_models(text)
        ret = extract_vrp_info(text)
        
        st.write(ret)