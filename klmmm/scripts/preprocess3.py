import os
import pandas as pd
import json
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # 여러 개의 공백을 하나로 줄임
    text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
    return text.strip()

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['context'] = data['context'].apply(lambda x: clean_text(x))  # 개행 문자 및 HTML 태그 제거

    if 'answers' not in data.columns:
        data['answers'] = ''

    def find_answer(row):
        try:
            answer_data = json.loads(row['answers'].replace("'", "\""))
            answer_text = answer_data.get('text', '')
            answer_start = answer_data.get('answer_start', 0)
            if answer_text in row['context']:
                return {'text': answer_text, 'answer_start': row['context'].find(answer_text)}
            else:
                return {'text': '', 'answer_start': 0}
        except json.JSONDecodeError:
            return {'text': '', 'answer_start': 0}

    data['answers'] = data.apply(find_answer, axis=1)
    return data

def preprocess_test_data(file_path):
    data = pd.read_csv(file_path)
    data['context'] = data['context'].apply(lambda x: clean_text(x))  # 개행 문자 및 HTML 태그 제거
    return data

def sliding_window(context, answer_text='', max_len=512, stride=256):
    tokens = context.split()
    windows = []
    for i in range(0, len(tokens), stride):
        window = " ".join(tokens[i:i+max_len])
        if answer_text in window:
            windows.append(window)
        if i + max_len >= len(tokens):
            break
    return windows

if __name__ == "__main__":
    # 파일 경로 설정
    train_file_path = '/content/drive/MyDrive/klmmm/data/train.csv'  # 실제 경로로 변경하세요
    test_file_path = '/content/drive/MyDrive/klmmm/data/test.csv'    # 실제 경로로 변경하세요
    output_dir = '/content/drive/MyDrive/klmmm/data/'

    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 데이터 전처리
    print("학습 데이터 전처리 중...")
    train_data = preprocess_data(train_file_path)
    processed_train_data = []
    for idx, row in train_data.iterrows():
        windows = sliding_window(row['context'], row['answers']['text'])
        for window in windows:
            new_row = row.copy()
            new_row['context'] = window
            processed_train_data.append(new_row)
    
    processed_train_data = pd.DataFrame(processed_train_data)
    processed_train_data['answers'] = processed_train_data['answers'].apply(json.dumps, ensure_ascii=False)
    processed_train_data.to_csv(os.path.join(output_dir, 'preprocessed_train3.csv'), index=False)
    print("학습 데이터 전처리 완료 및 저장")

    print("테스트 데이터 전처리 중...")
    test_data = preprocess_test_data(test_file_path)
    
    processed_test_data = []
    for idx, row in test_data.iterrows():
        windows = sliding_window(row['context'])
        for window in windows:
            new_row = row.copy()
            new_row['context'] = window
            processed_test_data.append(new_row)
    
    processed_test_data = pd.DataFrame(processed_test_data)
    processed_test_data.to_csv(os.path.join(output_dir, 'preprocessed_test3.csv'), index=False)
    print("테스트 데이터 전처리 완료 및 저장")
