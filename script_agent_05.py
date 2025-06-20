# %%
from typing import TypedDict
from urllib.parse import urlparse
import re
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from youtube_transcript_api import YouTubeTranscriptApi

# %%
load_dotenv()

# %%
llm = ChatOpenAI(model="gpt-4o", streaming=True)

# %%
class AgentState(TypedDict):
    """
    에이전트의 상태를 관리하는 객체입니다.

    Attributes:
        url (str): 사용자가 입력한 유튜브 URL
        transcript (str): 추출된 영상 자막 텍스트
        summary (str): LLM이 생성한 최종 요약 (JSON 형식)
        error (str): 처리 과정에서 발생한 오류 메시지
    """
    url: str
    transcript: str
    script_summary: str
    error: str

graph_builder = StateGraph(AgentState)

# %%
def extract_video_id(url):
    """YouTube URL에서 영상 ID를 추출하는 헬퍼 함수입니다."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    
    parsed_url = urlparse(url)
    if parsed_url.hostname == "googleusercontent.com" and parsed_url.path.startswith('/youtube.com/'):
        return parsed_url.path.split('/')[-1]
    
    return None

# %%
def get_youtube_transcript(state: AgentState) -> dict:
    """
    state에서 URL을 받아 자막을 추출하고, 
    결과를 state의 'transcript' 또는 'error' 필드에 저장합니다.
    """
    print("🚀 [Tool] get_youtube_transcript 호출됨")
    user_url = state["url"]
    
    try:
        video_id = extract_video_id(user_url)
        if not video_id:
            raise ValueError("유효한 유튜브 URL에서 Video ID를 추출할 수 없습니다.")

        print(f"✅ 영상 ID 추출 성공: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        transcript_text = " ".join([item['text'] for item in transcript_list])

        if len(transcript_text) < 100:
            raise ValueError("자막 내용이 너무 짧아 요약할 수 없습니다.")
        
        if len(transcript_text) > 15000:
            print(f"⚠️ 자막 크기 초과. 일부만 사용 (최대 15000자)")
            transcript_text = transcript_text[:15000]

        print("✅ 1. 자막 추출 성공")
        return {"transcript": transcript_text, "error": None}

    except Exception as e:
        error_message = f"ERROR: 자막 추출 중 오류 발생 - {e}"
        print(f"🚨 {error_message}")
        return {"transcript": "", "error": error_message}

# %%
summarize_prompt = (
    "너는 유튜브 영상의 스크립트(자막)를 분석해, 간결하고 보기 좋은 **JSON 형식 요약**을 작성하는 AI 전문가입니다.\n\n"

    "🔹 **목표**\n"
    "스크립트를 바탕으로 다음 항목들을 포함한 답변을 JSON 형식으로 작성하세요. 각 항목은 정확하고 친절한 말투로 작성하되, 너무 길지 않게 요약하세요.\n\n"

    "🔹 **출력 형식(JSON)**\n"
    "{\n"
    '  "요약": "영상의 핵심 내용을 줄바꿈 포함하여 부드럽게 설명",\n'
    '  "운동 강도": "예: 초급자용, 모든 레벨, 고강도 등",\n'
    '  "운동 루틴": [\n'
    '    "1. 🧘‍♀️ 동작 이름 - 간단한 설명",\n'
    '    "2. 🤲 동작 이름 - 간단한 설명",\n'
    "    ...\n"
    "  ],\n"
    '  "자극 신체 부위": "쉼표로 구분된 부위 목록 (ex. 어깨, 종아리, 허리)"\n\n'
    '  "영상에 대한 댓글 반응도 궁금하시다면 알려드릴게요!"\n'
    "}"

    "🔹 **작성 규칙**\n"
    "- 목록은 너무 길지 않게 핵심 위주로 요약하세요. 단, 주요 동작은 빠짐없이 포함해야 합니다.\n"
    "- 각 동작 설명은 짧고 이해하기 쉽게 표현하고, 말투는 딱딱하지 않게 하세요.\n"
    "- 반드시 마지막에 댓글 요약 제공 여부에 대한 질문을 포함하세요.\n"
    "- 불명확한 내용은 임의로 추측하지 말고, 스크립트에 기반하여 최대한 문맥적으로 유추하세요.\n"
    "- 챗봇 인터페이스에서 **한눈에 보기 좋게** 표현하세요.\n"
)

# %%
def summarize_transcript(state: AgentState) -> dict:
    """
    state의 'transcript' 필드 내용을 바탕으로 요약을 생성하고, 
    'summary' 필드를 업데이트합니다.
    """
    print("🚀 [Tool] summarize_transcript 호출됨")
    transcript = state["transcript"]
    
    prompt_messages = [
        SystemMessage(content=summarize_prompt),
        HumanMessage(content=f"[분석할 스크립트]\n---\n{transcript}\n---\n\n이 영상의 내용을 분석하여 필수 JSON 형식에 맞춰 요약해주세요.")
    ]
    try:
        response = llm.invoke(prompt_messages)
        script_summary = response.content
        print("✅ 2. 요약 생성 성공")
        return {"script_summary": script_summary}
    except Exception as e:
        error_message = f"ERROR: 요약 생성 중 오류 발생 - {e}"
        print(f"🚨 {error_message}")
        return {"script_summary": "", "error": error_message}

# %%
def route_after_transcript(state: AgentState) -> str:
    """
    state의 'error' 필드에 내용이 있는지 확인하여 다음 단계를 결정합니다.
    """
    if state.get("error"):
        print("🚨 오류가 감지되어 프로세스를 종료합니다.")
        return END
    else:
        print("✅ 스크립트 추출 성공. 요약 단계로 이동합니다.")
        return "summarize_transcript"

# %%
graph_builder.add_node("summarize_transcript", summarize_transcript)
graph_builder.add_node("get_youtube_transcript", get_youtube_transcript)

# %%
graph_builder.add_edge(START, "get_youtube_transcript")
graph_builder.add_conditional_edges(
    "get_youtube_transcript",
    route_after_transcript,
    {
        "summarize_transcript": "summarize_transcript",
        END: END
    }
)
graph_builder.add_edge("summarize_transcript", END)

# %%
graph = graph_builder.compile()
# graph

# %%
def run_agent(url: str):
    """
    에이전트를 실행하고 최종 결과를 출력합니다.
    - 초기 입력을 'url' 필드에 담아 전달합니다.
    - 최종 결과는 'comment_summary' 필드에서 가져옵니다.
    - 'error' 필드를 확인하여 오류를 처리합니다.
    """
    inputs = {"url": url}
    final_state = graph.invoke(inputs)

    if final_state.get("error"):
        print("\n" + "="*30)
        print("❌ 최종 실행 중 오류 발생:")
        print("="*30)
        print(final_state["error"])
        return

    content = final_state.get("script_summary", "")

    print("\n" + "="*30)
    print("✅ 최종 요약 (JSON 형식):")
    print("="*30)

    if not content or not content.strip():
        print("⚠️ 요약 결과가 비어 있습니다. content 값:", repr(content))
        return

    # LLM 응답에서 마크다운 코드 블록 제거
    if content.strip().startswith("```json"):
        start_index = content.find('{')
        end_index = content.rfind('}')
        if start_index != -1 and end_index != -1:
            content = content[start_index : end_index + 1]

    try:
        parsed_json = json.loads(content)
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("⚠️ JSON 파싱 실패. 원본 content 출력:")
        print(content)

# # %%
# test_url = "https://youtu.be/sLe6jgHoYtk?si=BP39AJQL1PvIoWBe"
# print(f"입력 URL: {test_url}\n---")
# run_agent(test_url)

# # %%



