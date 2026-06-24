import os
from openai import OpenAI
from dotenv import load_dotnev
from typing import List, Dict

load_dotnev()

class AgentsLLM:
    def __init__(self, model: str=None, apiKey: str=None, baseurl: str=None, timeout: int=None):

        self.model = model or os.getenv("MODEL_NAME")
        apiKey = apiKey or os.getenv("OPENAI_API_KEY")
        baseUrl = baseurl or os.getenv("OPENAI_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")
        
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)
    
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:

        print(f"正在调用大语言模型进行思考")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
                )
            
    