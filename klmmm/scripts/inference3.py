import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import argparse
import re

def refine_answer(answer):
    # 괄호 내용 제거
    answer = re.sub(r'\([^)]*\)', '', answer)

    # 필요한 특수문자(%.$℃₩€£¥ 등)를 제외한 특수문자 제거
    answer = re.sub(r'[^가-힣a-zA-Z0-9\s%.$℃₩€£¥~㎜㎾m○㎡\-\.\ㆍ]', '', answer)

    # 공백이 두 개 이상이면 하나로 줄임
    answer = re.sub(r'\s+', ' ', answer).strip()

    # 인용 표현을 감지
    quote_pattern = r'(["\'])(?:(?=(\\?))\2.)*?\1'
    quotes = re.findall(quote_pattern, answer)

    if quotes:
        # 인용 표현이 있는 경우, 인용 표현을 제외한 나머지 부분에서 조사 제거
        for quote in quotes:
            answer = answer.replace(quote[0], "")
        answer = re.sub(r'(은|는|이|가|을|를|에|의|이다|었다|하기|도|하게|한다|로|고|다|와|과|에는|으로|이며|으로서|로서|에서|로부터|해|이나|부터|이라|라고|까지만|까지|조차|뿐|만|라|라는|까지|으로의|엔|을까요|이었다|였다|다고|하는|합니다|한|기준으로)$', '', answer)
        # 인용 표현을 다시 삽입
        answer = " ".join(quotes) + answer
    else:
        # 인용 표현이 없는 경우, 전체에서 조사 제거
        answer = re.sub(r'(은|는|이|가|을|를|에|의|이다|었다|하기|도|하게|한다|로|고|다|와|과|에는|으로|이며|으로서|로서|에서|로부터|해|이나|부터|이라|라고|까지만|까지|조차|뿐|만|라|라는|까지|으로의|엔|을까요|이었다|였다|다고|하는|합니다|한|기준으로)$', '', answer)

    # 여러 개의 공백을 하나로 줄임
    answer = re.sub(r'\s+', ' ', answer).strip()

    # 로그 추가
    print(f"Refined answer: {answer}")

    return answer

def main(args):
    # 테스트 데이터 로드
    test_data = pd.read_csv(args.test_data_path)
    
    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    
    # 질의응답 파이프라인 설정
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    
    # 결과 리스트
    results = []

    # 테스트 데이터에 대해 질의응답 수행
    for index, row in test_data.iterrows():
        question = row["question"]
        context = row["context"]
        result = qa_pipeline(question=question, context=context)
        refined_answer = refine_answer(result['answer'])
        
        # 로그 추가: 원본 답변과 정제된 답변 출력
        print(f"Original answer: {result['answer']}")
        print(f"Refined answer: {refined_answer}")
        
        results.append({
            "id": row["id"],
            "answer": refined_answer
        })

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)
    
    # 결과를 sample_submission20.csv에 저장
    submission = pd.merge(test_data[['id']], results_df, on='id')
    submission.to_csv(args.submission_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the preprocessed test data CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--submission_path", type=str, required=True, help="Path to the submission CSV file")
    
    args = parser.parse_args()
    main(args)