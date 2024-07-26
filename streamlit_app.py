import os
import streamlit as st
import random
import shutil
import string
import yaml
import pdfplumber
from hugchat import hugchat
from hugchat.login import Login
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from HuggingChatAPI import HuggingChat
from langchain.embeddings import HuggingFaceHubEmbeddings
from promptTemplate import prompt4conversation, prompt4Context

# Configuration

DEFAULT_EMAIL = st.secrets["DEFAULT_EMAIL"]
DEFAULT_PASS = st.secrets["DEFAULT_PASS"]
DEFAULT_TOKEN = st.secrets["DEFAULT_TOKEN"]
PDF_FOLDER_PATH = st.secrets["PDF_FOLDER_PATH"]
CHROMA_DB_PATH = st.secrets["CHROMA_DB_PATH"]

# Initialize HuggingFaceHubEmbeddings
repo_id = "sentence-transformers/all-mpnet-base-v2"

# Streamlit page configuration
st.set_page_config(
    page_title="Talk with Everything💬", page_icon="🤗", layout="wide", initial_sidebar_state="expanded"
)
st.markdown('<style>.css-w770g5{width: 100%;}.css-b3z5c9{width: 100%;}.stButton>button{width: 100%;}.stDownloadButton>button{width: 100%;}</style>', unsafe_allow_html=True)

# Initialize session state variables if not already set
if 'hf_email' not in st.session_state:
    st.session_state['hf_email'] = DEFAULT_EMAIL
if 'hf_pass' not in st.session_state:
    st.session_state['hf_pass'] = DEFAULT_PASS
if 'hf_token' not in st.session_state:
    st.session_state['hf_token'] = DEFAULT_TOKEN
if 'hf' not in st.session_state:
    st.session_state['hf'] = HuggingFaceHubEmbeddings(
        repo_id=repo_id,
        task="feature-extraction",
        huggingfacehub_api_token=st.session_state['hf_token'],
    )  # type: ignore
if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] = None
if 'LLM' not in st.session_state:
    st.session_state['LLM'] = None
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello I am **JayGPT**, here to assist you about Jay. You can ask me anything about Jay professionally. How may I help you?"]
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']
if 'documents' not in st.session_state:
    st.session_state['documents'] = []
if 'pdf' not in st.session_state:
    st.session_state['pdf'] = None
if 'db' not in st.session_state:
    st.session_state['db'] = None
if 'cookies' not in st.session_state:
    st.session_state['cookies'] = None

# Function to log in and set up the chatbot
def login_and_setup():
    with st.spinner('🚀 Logging in and setting up...'):
        try:
            sign = Login(st.session_state['hf_email'], st.session_state['hf_pass'])
            cookies = sign.login()
            st.session_state['cookies'] = cookies
            st.session_state['chatbot'] = hugchat.ChatBot(cookies=cookies.get_dict())
            id = st.session_state['chatbot'].new_conversation()
            st.session_state['chatbot'].change_conversation(id)
            st.session_state['conversation'] = id
            st.session_state['LLM'] = HuggingChat(email=st.session_state['hf_email'],
                                                  psw=st.session_state['hf_pass'])
        except Exception as e:
            st.error(e)
            st.info("⚠️ Please check your credentials and try again.")
            st.error("⚠️ Don't abuse the API")
            st.warning("⚠️ If you don't have an account, you can register [here](https://huggingface.co/join).")
            del st.session_state['hf_email']
            del st.session_state['hf_pass']
            del st.session_state['hf_token']
            st.experimental_rerun()

# Automatically load documents from the default folder
def load_documents_from_folder(folder_path):
    documents = []
    with st.spinner('🔨 Reading documents from the default folder...'):
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                with pdfplumber.open(os.path.join(folder_path, filename)) as pdf:
                    documents += [page.extract_text() for page in pdf.pages]
    return documents

def setup_documents():
    if not os.path.exists(CHROMA_DB_PATH):
        st.session_state['documents'] = load_documents_from_folder(PDF_FOLDER_PATH)
        st.session_state['documents_loaded'] = True
        if st.session_state['documents']:
            # Split documents into chunks
            with st.spinner('🔨 Creating vectorstore...'):
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.create_documents(st.session_state['documents'])
                # Select embeddings
                embeddings = st.session_state['hf']
                # Create a vectorstore from documents
                db = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)

            with st.spinner('🔨 Saving vectorstore...'):
                # save vectorstore
                db.persist()
                # create .zip file of directory to download
                shutil.make_archive(CHROMA_DB_PATH, 'zip', CHROMA_DB_PATH)
                # save in session state and download
                st.session_state['db'] = f"{CHROMA_DB_PATH}.zip"

            with st.spinner('🔨 Creating QA chain...'):
                # Create retriever interface
                retriever = db.as_retriever()
                # Create QA chain
                qa = RetrievalQA.from_chain_type(llm=st.session_state['LLM'], chain_type='stuff',
                                                 retriever=retriever, return_source_documents=True)
                st.session_state['pdf'] = qa
    else:
        # Load existing Chroma database
        with st.spinner('🔨 Loading existing vectorstore...'):
            embeddings = st.session_state['hf']
            db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
            retriever = db.as_retriever()
            qa = RetrievalQA.from_chain_type(llm=st.session_state['LLM'], chain_type='stuff',
                                             retriever=retriever, return_source_documents=True)
            st.session_state['pdf'] = qa

# Perform initial setup if necessary
if st.session_state['cookies'] is None:
    login_and_setup()

if 'documents_loaded' not in st.session_state or not st.session_state['documents']:
    setup_documents()

# User input
# Layout of input/response containers
input_container = st.container()
response_container = st.container()
data_view_container = st.container()
loading_container = st.container()

## Applying the user input box
with input_container:
    input_text = st.chat_input("🧑‍💻 Write here 👇", key="input")

# with data_view_container:
#     if 'pdf' in st.session_state:
#         with st.expander("🤖 View your **DOCUMENTs**"):
#             st.write(st.session_state['documents'])

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    final_prompt = ""
    make_better = True
    source = ""

    with loading_container:

        if 'pdf' in st.session_state:
            # get only last message
            context = f"User: {st.session_state['past'][-1]}\nBot: {st.session_state['generated'][-1]}\n"
            with st.spinner('🚀 Using tool to get information...'):
                result = st.session_state['pdf']({"query": prompt})
                solution = result["result"]
                if len(solution.split()) > 110:
                    make_better = False
                    final_prompt = solution
                    if 'source_documents' in result and len(result["source_documents"]) > 0:
                        final_prompt += "\n\n✅Source:\n"
                        for d in result["source_documents"]:
                            final_prompt += "- " + str(d) + "\n"
                else:
                    final_prompt = prompt4Context(prompt, context, solution)
                    if 'source_documents' in result and len(result["source_documents"]) > 0:
                        source += "\n\n✅Source:\n"
                        for d in result["source_documents"]:
                            source += "- " + str(d) + "\n"

        else:
            # get last message if exists
            if len(st.session_state['past']) == 1:
                context = f"User: {st.session_state['past'][-1]}\nBot: {st.session_state['generated'][-1]}\n"
            else:
                context = f"User: {st.session_state['past'][-2]}\nBot: {st.session_state['generated'][-2]}\nUser: {st.session_state['past'][-1]}\nBot: {st.session_state['generated'][-1]}\n"

            final_prompt = prompt4conversation(prompt, context)

        if make_better:
            with st.spinner('🚀 Generating response...'):
                print(final_prompt)
                response = st.session_state['chatbot'].chat(final_prompt, temperature=0.5, top_p=0.95,
                                                            repetition_penalty=1.2, top_k=50,
                                                            max_new_tokens=1024)
                response += source
        else:
            print(final_prompt)
            response = final_prompt

    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if input_text and 'hf_email' in st.session_state and 'hf_pass' in st.session_state:
        response = generate_response(input_text)
        st.session_state.past.append(input_text)
        st.session_state.generated.append(response)

    # print message in normal order, first user then bot
    if 'generated' in st.session_state:
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                with st.chat_message(name="user"):
                    st.markdown(st.session_state['past'][i])

                with st.chat_message(name="assistant"):
                    if len(st.session_state['generated'][i].split("✅Source:")) > 1:
                        source = st.session_state['generated'][i].split("✅Source:")[1]
                        mess = st.session_state['generated'][i].split("✅Source:")[0]

                        st.markdown(mess)
                        # with st.expander("📚 Source of message number " + str(i + 1)):
                        #     st.markdown(source)

                    else:
                        st.markdown(st.session_state['generated'][i])

            st.markdown('', unsafe_allow_html=True)


    else:
        st.info("👋 Hey , we are very happy to see you here 🤗")
        st.info("👉 Please Login to continue, click on top left corner to login 🚀")
        st.error("👉 If you are not registered on Hugging Face, please register first and then login 🤗")
