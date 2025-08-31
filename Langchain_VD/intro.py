import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()

# llm = ChatOpenAI(model= 'gpt-4', temperature=1)
# res1 = llm.invoke("explain about Ai Agents in two lines or 20 words")
# res2 = llm.invoke("explain the Difference between Ai agents and LLMs in 10 words")
# print(res1.content)
# print(res2.content)


# -------------LLMChains(Without Output Parser)------------
# llm = ChatOpenAI(model= 'gpt-3.5-turbo', temperature=0.9)
# prompt = PromptTemplate(
#     input_variables=["Product_name"],
#     template="You are a helpful assistant who suggests 1 or 2 best companies which make {Product_name}."
# )
# # chain = LLMChain(llm=llm, prompt=prompt)
# chain = prompt | llm

# # print(chain.run("laptops"))
# print(chain.invoke("laptops").content)


# # ----------------Memory----------

# llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=ConversationBufferMemory()
# )

# # Start the conversation
# conversation.predict(input="Tell me about yourself.")

# # Continue the conversation
# conversation.predict(input="What can you do?")
# conversation.predict(input="How can you help me with data analysis?")

# # Display the conversation
# print(conversation)

# # ------------VectorDatabase(Since we don't have packages realted to DeepLake,this code will not work)----------------
# llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# # create our documents
# texts = [
#     "Napoleon Bonaparte was born in 15 August 1769",
#     "Louis XIV was born in 5 September 1638"
# ]
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.create_documents(texts)

# # create Deep Lake dataset
# # Todo: use your organization id here. (by default, org id is your username)
# my_activeloop_org_id = "venkateshtentu" 
# my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
# dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# # add documents to our Deep Lake dataset
# db.add_documents(docs)

# ----------------------------Agents in Langchain----------------------------
# Embeddings
from langchain_community.embeddings import OpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import DeepLake

# LLMs (use langchain_openai now)
from langchain_openai import OpenAI

# Agents / Tools
from langchain_community.agent_toolkits import load_tools

# Google Search
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain.agents import Tool

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)


search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    )
]

agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True,
                         max_iterations=6)

response = agent("What's the latest news about the Mars rover?")
print(response['output'])