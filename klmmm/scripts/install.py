from transformers import AutoTokenizer, AutoModelForCausalLM

# 로컬에 저장된 모델 경로 설정
model_directory = "/content/drive/MyDrive/klmmm/models/llama-3-1-8b"

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)
