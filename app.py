import streamlit as st
import os
from dotenv import load_dotenv
from llm import get_ai_message

st.set_page_config(
    page_title="유튜브 분석 챗봇",
    page_icon=":guardsman:",
)

st.title("유튜브 분석 챗봇")
st.caption("유튜브 영상의 스크립트 요약과 댓글 리포트를 생성해 드립니다.")

load_dotenv()

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="유튜브 URL을 입력하거나 분석 요청을 해보세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("AI가 답변을 생성하는 중입니다..."):
        ai_response_stream = get_ai_message(user_question)
        ai_message = ""
        for chunk in ai_response_stream:
            ai_message = chunk  # 마지막 chunk만 저장 (스트림이지만 대부분 1회)
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})