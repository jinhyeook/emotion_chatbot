import numpy as np
import cv2
import streamlit as st

from tensorflow.keras.models import load_model
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from PIL import Image
from contants import MODEL_PATH, API_KEY


class EmotionChatService:
    def __init__(self):
        try:
            self.model = load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"모델 로드에 실패했습니다: {e} 🚨")
            raise
        self.chatbot = self._setup_chatbot()
        self.songbot = self._setup_songbot()
        self.class_labels = {0: "화남", 1: "즐거움", 2: "당황함", 3: "슬픔"}

    def _setup_chatbot(self):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=API_KEY
        )
        chat_template = """
        너는 고도로 훈련된 심리상담사야. 사용자가 겪고 있는 감정을 정확히 파악하고, 이를 바탕으로 공감과 실질적인 도움을 제공하는 것이 너의 역할이야.
        
        다음 사항을 반드시 준수해야 해:
        1. 사용자를 "선생님"이라고 부르며 따뜻하고 진정성 있는 태도로 대화해.
        2. 대답은 간결하면서도 명확하고, 50자 내외로 작성하되 내용이 충실해야 해.
        3. 사용자의 질문에 적합한 감정 분석과 구체적인 조언을 제공해.
        4. 감정 상태에 따라 다음의 대화 톤을 유지해:
           - 슬픔: 위로와 안정을 주는 부드러운 어조.
           - 화남: 진정과 이해를 돕는 차분한 어조.
           - 당황함: 안심시키고 명확성을 제공하는 어조.
           - 즐거움: 긍정적이고 격려하는 어조.
        5. 심리적으로 의미 있는 해결책을 제안하고, 실질적인 도움을 줄 수 있도록 노력해.
        
        # 응답 예시
        Q: 너무 우울해요. 아무것도 하기 싫어요.
        A: 선생님, 이런 감정은 누구나 겪을 수 있어요. 잠시 쉬면서 자신을 돌봐주세요.

        이전 대화: {history}
        사용자의 입력: {input}
        답변:"""

        return ConversationChain(
            llm=llm,
            prompt=PromptTemplate(input_variables=["history", "input"], template=chat_template),
            memory=ConversationBufferMemory()
        )

    def _setup_songbot(self):
        song_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=API_KEY
        )

        song_template = """
        너는 전문 노래 추천 전문가야. 사용자의 감정 상태에 따라 적절한 한국 노래를 추천하고, 반드시 정확한 정보를 제공해야 해.
        
        다음 사항을 반드시 준수해야 해:
        1. 추천 노래는 반드시 "노래제목 - 가수이름" 형식으로 제공하고, 해당 가수가 발표한 곡이어야 해.
        2. 노래를 추천할 때 이유를 간략히 설명하되, 추천 이유가 감정 상태와 명확히 연관되어야 해.
        3. 감정 상태에 따라 다음 기준으로 추천 곡을 선택해:
           - 슬픔: 위로와 희망을 주는 발라드.
           - 화남: 감정을 해소할 수 있는 강렬하고 에너지 넘치는 곡.
           - 당황함: 안정감을 주는 차분한 멜로디.
           - 즐거움: 활기차고 긍정적인 분위기의 곡.
        4. 추천은 신뢰성을 기반으로 하며, 잘못된 매칭이 없도록 유의해.
        
        # 예시
        Q: 슬플 때 들을만한 노래 추천해줘.
        A: "비와 당신의 이야기 - 부활"은 슬픔을 위로하고 희망을 줄 수 있는 곡이에요.

        질문: {question}
        답변:"""

        return LLMChain(
            llm=song_llm,
            prompt=PromptTemplate(input_variables=["question"], template=song_template)
        )

    # 얼굴부분만 크롭
    def process_image(self, uploaded_file):
        try:
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                return None, "얼굴을 찾을 수 없습니다. 😢"

            x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
            y = max(y - int(h * 0.5), 0)
            h = int(h * 1.5)
            cropped_img = img_array[y:y+h, x:x+w]
            return self.predict_emotion(cropped_img), None
        except Exception as e:
            return None, f"이미지 처리 중 오류: {e} ⚠️"

    def predict_emotion(self, img_array):
        try:
            img = Image.fromarray(img_array).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = self.model.predict(img_array)
            return self.class_labels[np.argmax(predictions)]
        except Exception as e:
            return f"예측 오류: {e} ⚠️"
