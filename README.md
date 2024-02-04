# streaming-chatbot
Chatbot streaming responses over websockets

<h2>Setup</h2> 

1. Setup venv 

<h6>/path/to/python/3.9>= -m venv ./venv </h6>

2. Install python dependencies 

 <h6>   pip install --upgrade pip </h6>
  <h6>      CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install - r requirements.txt
 </h6>

3. Setup Local LLM 

<h6>mkdir ./models/ </h6>

<h6>Download LLAMA2 from <a href="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF">here </a> and save in ./models  </h6>
 <h6>
Create config.json
<code>
{
    "urls": [<List of URLS>]
    "model_path": ".models/<model_name>.gguf"
}
</code>
</h6>  

4. Deploy  
<code><h5>uvicorn main:app --host 0.0.0.0 --port 8000 --reload</h5></code>


   
