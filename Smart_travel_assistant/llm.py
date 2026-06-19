from openai import OpenAI

class OpenAICompatibleClient:

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str) -> str:

        print(f"正在调用大语言模型{self.model}...")

        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                stream = False
            )
            answer = response.choices[0].message.content
            print("大语言模型加载成功")
            return answer
        
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"