import pandas as pd
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, AutoModel
from transformers import DefaultDataCollator
import torch
from torch.utils.data import Dataset
import argparse
from sklearn.model_selection import train_test_split
import json
import evaluate
from transformers import EvalPrediction, EarlyStoppingCallback
import numpy as np
import os
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))
    torch.cuda.set_device(0)  # 첫 번째 GPU 사용

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 전역 변수로 선언
global augmentation_tokenizer, augmentation_model
augmentation_tokenizer = None
augmentation_model = None

def initialize_augmentation_model():
    global augmentation_tokenizer, augmentation_model
    augmentation_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    augmentation_model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data['answers'] = self.data['answers'].apply(self.ensure_valid_json)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        context = row['context']
        question = row['question']
        answer = json.loads(row['answers'])
        answer_text = answer['text']
        answer_start = answer['answer_start']

        encoding = self.tokenizer(
            context,
            question,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_offsets_mapping=True
        )

        offset_mapping = encoding.pop('offset_mapping')

        start_positions = []
        end_positions = []

        for i, (start, end) in enumerate(offset_mapping):
            if start <= answer_start < end:
                start_positions.append(i)
            if start < answer_start + len(answer_text) <= end:
                end_positions.append(i)

        if len(start_positions) == 0:
            start_positions = [0]
        if len(end_positions) == 0:
            end_positions = [0]

        encoding['start_positions'] = start_positions[0]
        encoding['end_positions'] = end_positions[-1]

        return {key: torch.tensor(val) for key, val in encoding.items()}

    def ensure_valid_json(self, answer):
        try:
            answer_json = json.loads(answer.replace("'", "\""))
            if 'text' in answer_json and 'answer_start' in answer_json:
                return json.dumps(answer_json, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        return json.dumps({'text': '', 'answer_start': 0}, ensure_ascii=False)

def get_synonyms(word, augmentation_tokenizer, augmentation_model, device, top_k=5):
    inputs = augmentation_tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = augmentation_model(**inputs)
    word_embedding = outputs.last_hidden_state.mean(dim=1)
    all_embeddings = augmentation_model.embeddings.word_embeddings.weight
    similarities = torch.cosine_similarity(word_embedding, all_embeddings)
    top_k_indices = similarities.argsort(descending=True)[:top_k+1]
    similar_words = [augmentation_tokenizer.decode([idx.item()]).strip() for idx in top_k_indices.cpu() if augmentation_tokenizer.decode([idx.item()]).strip() != word]
    return similar_words[:top_k]

def synonym_replacement(words, n, augmentation_tokenizer, augmentation_model, device):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        similar_words = get_synonyms(random_word, augmentation_tokenizer, augmentation_model, device)
        if similar_words:
            synonym = random.choice(similar_words)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words


def question_variation(question):
    variations = [
        f"다음 질문에 답해주세요: {question}",
        f"{question}에 대해 설명해주세요.",
        f"'{question}'에 대한 답변을 제공해주세요.",
        f"다음에 대해 알려주세요: {question}"
    ]
    return random.choice(variations)

def augment_data(data, augmentation_tokenizer, augmentation_model, device, augment_ratio=0.5):
    augmented_data = []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Augmenting data"):
        if random.random() < augment_ratio:
            try:
                context = row['context']
                question = row['question']
                answer = row['answers']
                words = question.split()

                # 동의어 대체
                augmented_question_1 = ' '.join(synonym_replacement(words, 1, augmentation_tokenizer, augmentation_model, device))
                
                # 질문 변형
                augmented_question_2 = question_variation(question)

                augmented_questions = [augmented_question_1, augmented_question_2]

                for aug_question in augmented_questions:
                    new_row = row.copy()
                    new_row['question'] = aug_question
                    augmented_data.append(new_row)

                if idx % 1000 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        else:
            augmented_data.append(row)

    augmented_data_df = pd.DataFrame(augmented_data)
    return augmented_data_df



def compute_metrics(p: EvalPrediction):
    print("Predictions shape:", [arr.shape for arr in p.predictions])
    print("Label IDs shape:", [arr.shape for arr in p.label_ids])
    
    metric = evaluate.load("squad_v2", trust_remote_code=True)
    start_logits, end_logits = p.predictions
    start_positions, end_positions = p.label_ids
    predictions = []
    references = []

    for i, (start_logit, end_logit, start_pos, end_pos) in enumerate(zip(start_logits, end_logits, start_positions, end_positions)):
        start_idx = np.argmax(start_logit)
        end_idx = np.argmax(end_logit)

        start_idx = max(0, min(start_idx, len(start_logit) - 1))
        end_idx = max(0, min(end_idx, len(end_logit) - 1))

        # 답변이 없는 경우를 처리합니다
        no_answer_prob = 1.0 - (start_logit[start_idx] + end_logit[end_idx])

        predictions.append({
            "id": str(i),
            "prediction_text": f"{start_idx}:{end_idx}",
            "no_answer_probability": float(no_answer_prob)
        })
        references.append({
            "id": str(i),
            "answers": {
                "text": [f"{start_pos}:{end_pos}"],
                "answer_start": [int(start_pos)]
            }
        })

    results = metric.compute(predictions=predictions, references=references)
    
    # 결과에서 사용 가능한 모든 메트릭을 반환합니다
    return {f"eval_{k}": v for k, v in results.items()}

class QATrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        print(f"Step: {self.state.global_step}, Loss: {logs.get('loss', 'N/A')}")

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        args = self.args
        model = self._wrap_model(self.model, training=False)
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = super().evaluation_loop
        try:
            output = eval_loop(
                dataloader,
                description,
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        if self.compute_metrics is not None:
            output.metrics.update(self.compute_metrics(EvalPrediction(predictions=output.predictions, label_ids=output.label_ids)))
        return output


def main(args):
    
    global augmentation_tokenizer, augmentation_model

    # 여기에 GPU 사용 가능 여부 확인 코드 추가
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # 증강 모델 초기화 및 디바이스로 이동
    print("Initializing augmentation model...")
    augmentation_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    augmentation_model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
    augmentation_model = augmentation_model.to(device)
    augmentation_model.to(device)

    print(f"Augmentation tokenizer initialized: {augmentation_tokenizer is not None}")
    print(f"Augmentation model initialized: {augmentation_model is not None}")

    # Load and split data
    try:
        data = pd.read_csv(args.train_data_path)
        print(data.head())
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    try:
        train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return

    # Load model and tokenizer for QA
    model_name = "monologg/koelectra-base-v3-finetuned-korquad"
    qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Prepare dataset
    train_dataset = QADataset(train_data, qa_tokenizer)  # qa_tokenizer로 변경
    val_dataset = QADataset(val_data, qa_tokenizer)  # qa_tokenizer로 변경

    data_collator = DefaultDataCollator(return_tensors="pt")
    
    

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        logging_dir='./my_logs',
        logging_steps=50,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        lr_scheduler_type='linear',
        fp16=True,  # fp16 정밀도 사용
        dataloader_num_workers=2,
        remove_unused_columns=True,
        disable_tqdm=False,
        report_to="tensorboard",
        # use_cuda 대신 no_cuda 사용
        no_cuda=False  # GPU 사용 (True로 설정하면 CPU 사용)
    )

    writer = SummaryWriter(log_dir=training_args.logging_dir)
    
    trainer = QATrainer(
        model=qa_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=qa_tokenizer,  # qa_tokenizer로 변경
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # main 함수 내에서
    print(f"Original train data size: {len(train_data)}")
    train_data = augment_data(train_data, augmentation_tokenizer, augmentation_model, device, augment_ratio=args.augment_ratio)
    print(f"Augmented train data size: {len(train_data)}")

    train_dataset = QADataset(train_data, qa_tokenizer)

    del augmentation_model
    del augmentation_tokenizer
    torch.cuda.empty_cache()

    # 체크포인트 폴더 생성
    os.makedirs("./checkpoints", exist_ok=True)

    # 가장 최근 체크포인트 찾기
    latest_checkpoint = None
    if os.path.exists("./checkpoints"):
        checkpoints = [f for f in os.listdir("./checkpoints") if f.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join("./checkpoints", latest_checkpoint)

    # 훈련 시작 또는 재개
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.train()

    print(f"Is model on GPU? {next(qa_model.parameters()).is_cuda}")
    
    # Save model
    trainer.save_model(args.model_save_path)
    qa_tokenizer.save_pretrained(args.model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the preprocessed train data CSV file")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--no_cuda", action='store_true', help="Use CPU instead of GPU")
    parser.add_argument("--augment_ratio", type=float, default=0.5, help="Ratio of data to augment")
    
    args = parser.parse_args()
    main(args)
