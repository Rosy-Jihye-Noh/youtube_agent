import streamlit as st
from youtube_agent import graph, graph_memory, continue_with_memory, memory
import uuid
import json

def _clean_and_parse_json(content: str) -> dict:
    """LLM의 응답에서 마크다운을 제거하고 JSON으로 파싱합니다."""
    if content.strip().startswith("```json"):
        start_index = content.find('{')
        end_index = content.rfind('}')
        if start_index != -1 and end_index != -1:
            content = content[start_index : end_index + 1]
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 파싱 실패 시 원본 텍스트나 에러 메시지를 포함한 객체를 반환할 수 있습니다.
        return {"original_content": content}

# 세션 상태에 thread_id 없으면 새로 생성
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# config에 넣기
config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.write(f"현재 thread_id: {st.session_state.thread_id}")

st.title("YouTube 요약 에이전트 (스크립트 + 댓글)")

# 1) URL 입력 받기
url_input = st.text_input("YouTube 영상 URL을 입력하세요", "")

if url_input:
    initial_state = {"url": url_input}
    
    if st.button("스크립트 요약 요청"):
        with st.spinner("스크립트 요약 중..."):
            step1_state = graph.invoke(initial_state, config=config)
            script_summary = _clean_and_parse_json(step1_state.get("script_summary"))
            if script_summary:
                st.markdown("### 📄 스크립트 요약 결과")
                st.json(script_summary)
            else:
                st.warning("스크립트 요약 결과가 없습니다.")
            
            # memory.set_state(step1_state, config)  # <-- 이 줄 삭제
    
    st.markdown("---")
    reply_input = st.text_input("댓글 요약을 원하시면 여기에 답변을 입력하세요 (예: 응, 네, 보여줘 등)")

    if reply_input:
        previous_state = initial_state
        
        update_state = {"reply": reply_input, "url": previous_state.get("url")}
        
        if st.button("댓글 요약 요청"):
            with st.spinner("댓글 요약 중..."):
                step2_state = continue_with_memory(graph, initial_state, config, update_state)
                comment_summary = step2_state.get("comment_summary")
                if comment_summary:
                    st.markdown("### 💬 댓글 요약 결과")
                    st.write(comment_summary)
                else:
                    st.warning("댓글 요약 결과가 없습니다.")

print(dir(memory))
