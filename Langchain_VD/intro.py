import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(temperature=0)
res = llm.invoke("What would be a good company name for a company that makes colorful socks?")
print(res.content)