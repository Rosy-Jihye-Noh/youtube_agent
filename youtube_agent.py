# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from typing import TypedDict, List, Optional, Dict

# %%
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# %%
class AgentState(TypedDict, total=False):
    url: str
    reply: Optional[str]
    script_summary: Optional[Dict[str, str | List[str]]]
    comment_summary: Optional[str]

graph_builder = StateGraph(AgentState)

# %%
# 라우터 설정 
class Route(BaseModel):
    target: Literal['summarize_script', 'summarize_comment'] = Field(
        description="The target for the query to answer"
    )

# 라우터 프롬프트 설정 
router_system_prompt = """
You are an expert at routing a user's message given these variables:
- URL: the YouTube video link (may be empty if none)
- 질문: the user's initial question or query

Classify a user's input into one of two categories: 'summarize_script' or 'summarize_comment'.

- If the message **does not contain a state[reply]** or **has blank state[reply]** (e.g., includes 'youtube.com' or 'youtu.be'), always route it to 'summarize_script'. 

- If the user's message is a **positive or affirmative response to a question like** "영상에 대한 댓글 반응도 궁금하시다면 알려드릴게요!" — such as "응", "네", "보여줘", "궁금해", etc., then route it to 'summarize_comment'.

- If the user does not respond or responds **negatively** (e.g., "괜찮아", "글쎄", "아니"), **do not route anywhere**. This means the flow should end silently.
"""

# 라우터 프롬프트 템플릿 설정
router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("user", "URL: {url}\n사용자 답변: {reply}")
])

# 라우터 형식으로 구조화한 llm 설정
structured_router_llm = llm.with_structured_output(Route)

# %%
def router(state: AgentState) -> Literal['summarize_script', 'commentsummarize_comment_summary']:
    """
    주어진 state에서 쿼리를 기반으로 적절한 경로를 결정합니다

    Args:
        state (AgentState): 에이전트의 현재 상태를 나타내는 딕셔너리

    Returns:
        AgentState: Literal['summarize_script', 'summarize_comment']: 쿼리에 따라 선택된 경로를 반환합니다.
    """

    url = state.get('url', '')
    reply = state.get('reply', '')
    
    router_chain = router_prompt | structured_router_llm
    route = router_chain.invoke({"url": url, "reply": reply})

    return route.target

# %%
from script_agent_05 import graph as script_agent
from comment_agent_05 import graph as comment_agent

# %%
graph_builder.add_node('summarize_script', script_agent)
graph_builder.add_node('summarize_comment', comment_agent)

# %%
graph_builder.add_conditional_edges(
    START,
    router,
    {   # 리턴값 :  노드이름 
        'summarize_script': 'summarize_script',
        'summarize_comment': 'summarize_comment',
    }
)

graph_builder.add_edge('summarize_script', END)
graph_builder.add_edge('summarize_comment', END)

# %%
graph = graph_builder.compile()
graph

# %%
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

graph_memory = graph_builder.compile(checkpointer=memory)


# %%
def continue_with_memory(graph, initial_state, config: dict, update: dict):
    previous_state = initial_state
    new_state = {**previous_state, **update}
    return graph.invoke(new_state, config=config)

# %%
# config = {"configurable": {"thread_id": "1"}}
# url = 'https://youtu.be/sLe6jgHoYtk?si=BP39AJQL1PvIoWBe'


# # %%
# initial_state = {'url': url}
# step1_state = graph.invoke(initial_state, config=config)
# print("📄 스크립트 요약 결과:")
# print(step1_state.get("script_summary", ""))

# # %%
# step2_state = continue_with_memory(graph, graph_memory, config, {"reply": "응", "url": step1_state.get("url")})
# print("\n💬 댓글 요약 결과:")
# print(step2_state.get("comment_summary", ""))


