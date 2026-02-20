import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os

#page config
st.set_page_config(page_title="Simple Langchain Chatbot with Groq", page_icon=":speech_balloon:")

#title
st.title("Simple Langchain Chatbot with Groq")
st.markdown("This is a simple chatbot built using Langchain and Groq. It can answer questions about the current date and time.")

with st.sidebar:
    st.header("Settings")

    #api key
    api_key = st.text_input("Groq API Key", type="password", help="Get free api key at console.groq.com")

    #model selection
    model_name = st.selectbox(
        "Select Model", 
        ["llama-3.1-8b-instant"],
        index=0
    )

    #clear button
    if st.button("clear chat"):
        st.session_state.messages=[]
        st.rerun()

#initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#initialize LLM
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None
    
    #initialize the Groq model
    llm= ChatGroq(groq_api_key=api_key, 
                  model_name=model_name,
                  temperature=0.7,
                  streaming=True)
    
    #Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),
            ("user", "{question}")
        ]
    )

    #create chain
    chain = prompt | llm | StrOutputParser()
    return chain

#get the chain
chain = get_chain(api_key, model_name)

if not chain:
    st.warning("Please enter your Groq API key in the sidebar to start chatting.")
    st.markdown("[Get your free API key here](https://console.groq.com/keys)")

else:
    #display the chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    #chat input
    if question:= st.chat_input("Ask me anything"):
        # add user message to session state
        st.session_state.messages.append({"role":"user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        #generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                #stream response from Groq
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")  # Add a cursor to indicate streaming

                message_placeholder.markdown(full_response)

                #add to history
                st.session_state.messages.append({"role":"assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")

#footer
st.markdown("---")
st.markdown("Built with ❤️ using [Langchain](https://python.langchain.com/) and [Groq](https://www.groq.com/)")