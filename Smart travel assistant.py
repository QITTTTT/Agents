import os
import requests
from tavily import TavilyClient

def get_weather(city: str) -> str:

    url = f"https://wttr.in/{city}?format=j1"

    try:
        reponse = requests.get(url)
        reponse.raise_for_status()

        data = reponse.json()

        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        return f"{city}当前天气:{weather_desc}, 气温{temp_c}摄氏度"
    
    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"
    
def get_attraction(city: str, weather: str) -> str:

    tavily = TavilyClient(api_key="sk-9GQz7e5P380xsfcxB47d0013319e42Bf83E361F04cBcC410")

    query = f"{city} 在 {weather} 天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        if response.get("answer"):
            return response["answer"]

        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)
    
    except Exception as e:
        
    # sk-9GQz7e5P380xsfcxB47d0013319e42Bf83E361F04cBcC410