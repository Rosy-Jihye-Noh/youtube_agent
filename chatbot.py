import os 
import httpx
import requests
import asyncio
from transformers import pipeline, AutoTokenizer
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "google/flan-t5-small"
text_generator = pipeline("text2text-generation", model=MODEL_NAME, device=-1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

OLLAMA_API_URL = "http://192.168.2.6:11434/api/chat"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def truncate_text_tokenwise(text: str, max_tokens: int = 1024) -> str:
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, clean_up_tokenization_spaces=True)


class Graph:
    def __init__(self):
        self.nodes = {}

    def add_nodes(self, *nodes):
        for node in nodes:
            self.nodes[node.name] = node

    def add_edge(self, from_node, to_node, output_key=None):
        # 현재 사용 안 함 (필요하면 구현)
        pass

    async def run_node(self, node_name, *args, **kwargs):
        node = self.nodes.get(node_name)
        if node is None:
            raise Exception(f"Node '{node_name}' not found in graph")
        if asyncio.iscoroutinefunction(node.run):
            return await node.run(*args, **kwargs)
        else:
            return node.run(*args, **kwargs)


class AskModelNode(BaseModel):
    name: str

    async def run(self, prompt: str, model_name: str) -> str:
        messages = [
            {"role": "system", "content": "You are a friendly health trainer chatbot. Respond in English."},
            {"role": "user", "content": prompt}
        ]
        payload = {"model": model_name, "messages": messages, "stream": False}
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(OLLAMA_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")


class CombineAnswerNode(BaseModel):
    name: str

    def run(self, answer1: str, answer2: str) -> str:
        prompt = (
            "Combine these two health advice answers into one fluent English answer:\n\n"
            "Answer 1:\n" + answer1 + "\n\nAnswer 2:\n" + answer2 + "\n\nCombined answer:"
        )
        prompt = truncate_text_tokenwise(prompt, max_tokens=1024)
        result = text_generator(prompt, max_new_tokens=1024, do_sample=False)
        return result[0]['generated_text']

def translate_en_to_ko(text: str) -> str:
    url = "https://api.mymemory.translated.net/get"
    params = {"q": text, "langpair": "en|ko"}
    resp = requests.get(url, params=params)
    return resp.json().get("responseData", {}).get("translatedText", "")

class TranslateNode(BaseModel):
    name: str
    def run(self, eng: str) -> str:
        return translate_en_to_ko(eng)

class GenerateYoutubeQueryNode(BaseModel):
    name: str

    def run(self, combined_answer: str) -> str:
        prompt = (
            "Based on the following health advice, generate a short and specific YouTube search keyword in English. "
            "Keep it under 6 words and only include exercises or stretches:\n\n"
            f"{truncate_text_tokenwise(combined_answer, 256)}\n\nSearch query:"
        )
        result = text_generator(prompt, max_new_tokens=16, do_sample=False)
        query = result[0]['generated_text'].strip()

        # 필터링: 쓸데없는 문장 제거
        if len(query.split()) > 8 or "I'm" in query or "chat" in query:
            return "stretches for neck pain"  # fallback query
        return query

class YoutubeSearchNode(BaseModel):
    name: str

    def run(self, query: str) -> str:
        params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "type": "video",
            "maxResults": 1,
            "videoEmbeddable": "true"
        }
        response = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items:
            return "No video found."
        video_id = items[0]["id"]["videoId"]
        return f"https://www.youtube.com/watch?v={video_id}"


class OutputNode(BaseModel):
    name: str

    def run(self, korean_answer: str, youtube_url: str, search_phrase: str) -> dict:
        print("=== 최종 한국어 답변 ===")
        print(korean_answer.strip())
        print("\n=== 추천 유튜브 동영상 ===")
        print(youtube_url)
        print("\n=== 유튜브 검색어 ===")
        print(search_phrase.strip())
        return {
            "korean_answer": korean_answer.strip(),
            "youtube_url": youtube_url,
            "search_phrase": search_phrase.strip()
        }


graph = Graph()

ask_llama = AskModelNode(name="ask_llama")
ask_gemma = AskModelNode(name="ask_gemma")
combine = CombineAnswerNode(name="combine")
translate = TranslateNode(name="translate")
gen_query = GenerateYoutubeQueryNode(name="generate_query")
youtube_search = YoutubeSearchNode(name="youtube_search")
output = OutputNode(name="output")

graph.add_nodes(ask_llama, ask_gemma, combine, translate, gen_query, youtube_search, output)


async def main():
    prompt = "나 등이 너무 아파. 스트레칭 알려줄 수 있어? 등이 뻐근해"
    prompt = truncate_text_tokenwise(prompt, max_tokens=512)

    llama_task = graph.run_node("ask_llama", prompt, "llama3")
    gemma_task = graph.run_node("ask_gemma", prompt, "gemma")

    answer1, answer2 = await asyncio.gather(llama_task, gemma_task)

    combined_answer = combine.run(answer1, answer2)
    korean_answer = translate.run(combined_answer)
    search_phrase = gen_query.run(combined_answer)
    youtube_url = youtube_search.run(search_phrase)

    output.run(korean_answer, youtube_url, search_phrase)

if __name__ == "__main__":
    asyncio.run(main())
