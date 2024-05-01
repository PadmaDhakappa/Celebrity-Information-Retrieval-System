#Integrating our code with OpenAI API

import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

os.environ["OPENAI_API_KEY"]=openai_key
#Streamlit Framework

st.title('Celebrity Search Results')
input_text=st.text_input("Search the topic u want")

#PromptTemplate

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template='Tell me about celebrity{name}'

)

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template='When was {person} born'

)


third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world"
)

#OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain1=LLMChain(llm=llm,prompt=first_input_prompt, verbose=True,output_key='person')
chain2=LLMChain(llm=llm,prompt=second_input_prompt, verbose=True,output_key='dob')

chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description')
parent_chain=SequentialChain(
    chains=[chain1,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)


#parent_chain=SimpleSequentialChain(chains=[chain1,chain2],verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))