import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from schemas import ChatResponse
from callback import StreamingLLMCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from websockets.exceptions import ConnectionClosedOK
from query_data import get_chain
import json

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
templates = Jinja2Templates(directory="templates")
app = FastAPI()

urls =[
        'https://python.langchain.com/docs/modules/data_connection/',
        'https://python.langchain.com/docs/modules/data_connection/document_loaders/'
    ]

def read_urls(config_path: str = 'config.json'):
    """
    Read urls from the config file
    """
    with open(config_path) as f:
        data = json.load(f)
    return data['urls']

def setup_retriever(path: str=None):
    """
    Function to setup the retriever using faiss,huggingface and webbase loader
    """
    ## 
    embeddings = HuggingFaceEmbeddings()
    if os.path.exists('url_db/'):
        db = FAISS.load_local("url_db/", embeddings=embeddings)
        return db
    else:
        loader = WebBaseLoader(urls)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        # embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"  
        
        db = FAISS.from_documents(docs, embeddings)
        return db

def search_retriever(query:str):
    """
    Function to search the retriever
    """
    search = db.similarity_search(query, k=2)
    return search

logging.info("Setting up retriever...")
db = setup_retriever(path="url_db/")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/test")
async def test_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    logging.info("WebSocket listening...")
    await websocket.accept()
    logging.info("WebSocket accepted...")
    stream_handler = StreamingLLMCallbackHandler(websocket)
    qa_chain = get_chain(stream_handler)
    while True:
        try:
            # Receive and send back the client message
            user_msg = await websocket.receive_text()
            logging.info(f"Received message from user....")
            resp = ChatResponse(sender="human", message=user_msg, type="stream")
            await websocket.send_json(resp.dict())
            
            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            search = search_retriever(user_msg)
            # Send the message to the chain and feed the response back to the client
            output = await qa_chain.ainvoke(
                {
                    "input": user_msg,
                    "context": search
                }
            )

            # Send the end-response back to the client
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("WebSocketDisconnect")
            # TODO try to reconnect with back-off
            break
        except ConnectionClosedOK:
            logging.info("ConnectionClosedOK")
            # TODO handle this?
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())
