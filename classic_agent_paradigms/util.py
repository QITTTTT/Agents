import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

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
            print("LLM响应成功")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)

            print()
            return "".join(collected_content)
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return None

if __name__ == '__main__':
    try:
        llmClient = AgentsLLM()

        exampleMessages = [
            {"role": "sustem", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
            ]
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
