import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# APIトークンをSecretsから取得
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=0.7,
    max_new_tokens=100
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="以下の質問に答えてください：{question}"
)

chain = prompt | llm

st.title("日本語質問応答チャットボット")

question = st.text_input("質問を入力してください:")

if question:
    response = chain.invoke({"question": question})
    st.write("回答:", response)
