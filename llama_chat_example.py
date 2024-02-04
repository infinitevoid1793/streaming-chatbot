from langchain.chains import LLMChain,StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import LlamaCpp
from os.path import expanduser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
import logging

logging.basicConfig(level=logging.INFO)


model_path = expanduser('/Users/sharangbhat/Code/OpenAI/models/llama-2-7b-chat.Q6_K.gguf')

if __name__ == '__main__':

    urls =[
        'https://python.langchain.com/docs/modules/data_connection/',
        'https://python.langchain.com/docs/modules/data_connection/document_loaders/'
    ]
    ## 
    loader = WebBaseLoader(urls)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    
    # embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"  
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Make sure the model path is correct for your system!
    llm = LlamaCpp(
    model_path=model_path,
    temperature=0,
    n_gpu_layers = -1,
    n_batch = 512,
    max_tokens=1000,
    n_ctx =4096,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
    embeddings =HuggingFaceEmbeddings()
    # db = FAISS.from_documents(docs, embeddings)
    
    # db.save_local("url_db")
    db = FAISS.load_local("url_db/", embeddings=embeddings)
    retriever = db.as_retriever()
    query = "wWhat is the purpose of document loaders?"
    search = db.similarity_search(query, k=2)

    template = """
    <s>[INST] <<SYS>>
    You are a helpful assistant. You will refer to the provided context which is a reference to a some url's and their content. You will provide the user with the most relevant information from the context. If you do not know the answer to a question, you truthfully say you do not know. But if you do know the answer, you provide it without any hesitation.
    Do not mentioned the above text in the response. The system prompt is meant only for you.
    Context: {context}
    <</SYS>>

    Question: {question} [/INST]
    """
 
    prompt = PromptTemplate(
        template=template,input_variables=["context","question"]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,input_key="question")
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    
    llm_chain.invoke(input={"question":"Hello!", "context":search})
