import os

# GPU 설정
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 모델 경로 및 API 키
MODEL_PATH = r"model/emotion_recognition_model.h5"
API_KEY = "Your_API_KEY"
