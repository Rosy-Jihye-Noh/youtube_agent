from comment_analysis import graph as comment_agent
from script_agent_02 import script_agent

def get_ai_message(user_message: str):
    """
    유튜브 URL이 포함되고 '스크립트' 또는 '요약'이 들어가면 스크립트 요약,
    아니면 댓글 리포트로 분기.
    """
    # 간단한 분기
    if ("스크립트" in user_message or "요약" in user_message):
        script_input = {"messages": [{"role": "user", "content": user_message}]}
        final_script_state = script_agent.invoke(script_input)
        summary = final_script_state.get('summary', '요약 결과가 없습니다.')
        yield summary
    else:
        from langchain_core.messages import HumanMessage
        comment_input = {"messages": [HumanMessage(content=user_message)]}
        final_comment_report = None
        for chunk in comment_agent.stream(comment_input, stream_mode='values'):
            final_comment_report = chunk['messages'][-1].content
            yield final_comment_report