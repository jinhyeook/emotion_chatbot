from service import EmotionChatService
from session import initialize_session_state
import streamlit as st


def main():
    service = EmotionChatService()
    initialize_session_state()

    st.title("🤣😡😱😭")
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("얼굴 사진을 업로드해주세요 📸", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            emotion, error = service.process_image(uploaded_file)
            if error:
                st.error(error)
            else:
                st.session_state.emotion = emotion
                st.success(f"감지된 감정: {emotion} 😊")
                st.image(uploaded_file, caption="업로드된 이미지", use_container_width=True)

                if not st.session_state.initial_question_asked:
                    emotion_questions = {
                        "슬픔": "나 지금 너무 슬퍼. 무슨 일이 있었는지 물어봐줘. 😢",
                        "화남": "나 지금 너무 화나. 무슨 일이 있었는지 물어봐줘. 😡",
                        "당황함": "나 지금 너무 당황스러워. 무슨 일이 있었는지 물어봐줘. 😨",
                        "즐거움": "나 지금 너무 즐거워. 무슨 일이 있었는지 물어봐줘. 😄",
                    }
                    initial_question = emotion_questions.get(emotion)
                    if initial_question:
                        response = service.chatbot.run({"input": initial_question})
                        st.session_state.chat_history.append(("챗봇", response))
                        st.session_state.initial_question_asked = True

    with col2:
        st.header("💬   대화하기")
        for idx, (role, message) in enumerate(st.session_state.chat_history):
            st.text_area(f"{role}:", message, height=70, disabled=True, key=f"{role}_{idx}")

        with st.form(key="message_form", clear_on_submit=True):
            user_input = st.text_input("메시지를 입력하세요 (종료하려면 'exit' 입력) 🧐", key="user_input")
            submitted = st.form_submit_button("전송")

        if submitted:
            if not st.session_state.emotion:
                st.warning("📸 먼저 사진을 업로드해주세요!")
            elif user_input.lower() == "exit":
                song_question = f"{st.session_state.emotion}일때 들을만한 한국노래 추천해줘."
                song_recommendation = service.songbot.run({"question": song_question})
                st.session_state.chat_history.append(("챗봇", f"노래 추천 🎵: {song_recommendation}"))
            elif user_input:
                st.session_state.chat_history.append(("사용자", user_input))
                response = service.chatbot.run({"input": user_input})
                st.session_state.chat_history.append(("챗봇", response))


if __name__ == "__main__":
    main()
