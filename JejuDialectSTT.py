import os
import json
import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import soundfile as sf
import wandb
from transformers import DataCollatorWithPadding
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from jiwer import wer
from transformers import IntervalStrategy


wandb.init(project="jejudialectstt")

torch.cuda.empty_cache()

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available, using CPU.")

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6"

# Step 1: 데이터 로드 및 전처리
def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["dataset"]

def preprocess_data(dataset):
    # Dataset의 구조를 통일
    audio_paths = []
    texts = []
    for item in dataset:
        audio_paths.append(item["audio_filepath"])
        texts.append(item["text"])
    return {"audio_filepath": audio_paths, "text": texts}

# JSON 파일 경로
json_file_path = "/home/aix23606/seoah/JejuDialectSTT/data_sample/dataset.json"
raw_data = load_data(json_file_path)
processed_data = preprocess_data(raw_data)

# Load the dataset into Hugging Face Dataset format
hf_dataset = Dataset.from_dict(processed_data)

# Step 2: 음성 파일 처리 함수 정의
def speech_file_to_array_fn(batch):
    try:
        speech_array, sampling_rate = sf.read(batch["audio_filepath"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
    except Exception as e:
        print(f"Error reading file {batch['audio_filepath']}: {e}")
        raise e
    return batch

# Apply audio preprocessing
hf_dataset = hf_dataset.map(speech_file_to_array_fn)

print("Full dataset length:", len(hf_dataset))

# Step 3: Dataset 분할
try:
    dataset_split = hf_dataset.train_test_split(test_size=0.1)
    train_valid_dataset = dataset_split['train']
    test_dataset = dataset_split['test']

    train_valid_split = train_valid_dataset.train_test_split(test_size=0.1)
    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']

except Exception as e:
    print(f"Error during dataset split: {e}")
    raise e

# Step 4: Processor 로드
processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

def prepare_dataset(batch):
    try:
        # 음성을 모델이 요구하는 형식으로 변환
        input_values = processor(batch["speech"], sampling_rate=batch["sampling_rate"], return_tensors="pt",
                                 padding=True).input_values

        # 라벨 처리
        labels = processor.tokenizer(batch["text"], return_tensors="pt", padding=True).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100

        batch["input_values"] = input_values[0]
        batch["labels"] = labels[0]
    except Exception as e:
        print("Error in prepare_dataset:", e)
        raise e
    return batch

try:
    train_dataset = train_dataset.map(prepare_dataset,
                                      remove_columns=["audio_filepath", "speech", "sampling_rate", "text"])
    valid_dataset = valid_dataset.map(prepare_dataset,
                                      remove_columns=["audio_filepath", "speech", "sampling_rate", "text"])
    test_dataset = test_dataset.map(prepare_dataset,
                                    remove_columns=["audio_filepath", "speech", "sampling_rate", "text"])
except Exception as e:
    print(f"Error during dataset preparation: {e}")
    raise e

# Step 5: 모델 로드
model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
model.freeze_feature_encoder()

# Step 6: 학습 설정
training_args = TrainingArguments(
    output_dir="./stt_model",
    group_by_length=True,
    per_device_train_batch_size=4,
    evaluation_strategy=IntervalStrategy.EPOCH,  # 매 epoch 마다 평가
    logging_strategy=IntervalStrategy.EPOCH,    # 매 epoch 마다 train loss 기록
    num_train_epochs=10,
    fp16=False,
    max_grad_norm=1.0,
    save_steps=1000,
    learning_rate=1e-5,
    warmup_steps=500,
    save_total_limit=2,
    gradient_accumulation_steps=2,
    dataloader_num_workers=2,
    report_to="wandb",
    gradient_checkpointing=False
)


model = model.to("cuda")

# DataCollatorCTCWithPadding 클래스 구현
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        label_features = []
        for feature in features:
            if isinstance(feature["labels"], torch.Tensor):
                label_features.append(feature["labels"])
            else:
                label_features.append(torch.tensor(feature["labels"]))

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = torch.nn.utils.rnn.pad_sequence(label_features, batch_first=True, padding_value=-100)
        batch["labels"] = labels_batch

        return batch

# DataCollator 변경
data_collator = DataCollatorCTCWithPadding(processor=processor)

# Trainer 설정
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor,
)

# Step 7: 학습 시작
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    raise e

# 학습 중간에 GPU 상태 출력
torch.cuda.synchronize()
print(torch.cuda.memory_summary())

# Step 8: 모델 저장
model.save_pretrained("./stt_model")
processor.save_pretrained("./stt_model")

print("Training complete and model saved!")

# Step 9: 테스트 데이터로 예측 확인 및 WandB 로깅
def evaluate_and_log_test_data(test_dataset):
    test_results = []
    correct = 0

    for idx, test_sample in enumerate(test_dataset):
        input_values = torch.tensor(test_sample["input_values"]).unsqueeze(0).to("cuda")
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        ground_truth = processor.decode(test_sample["labels"], skip_special_tokens=True)

        # 빈 문자열 예외 처리
        if not prediction:
            prediction = "<empty>"

        # 정확도 계산
        if prediction == ground_truth:
            correct += 1

        # 파일명 추출
        file_name = test_sample.get("audio_filepath", f"sample_{idx}")

        # 결과 저장
        test_results.append({
            "file": file_name,
            "ground_truth": ground_truth,
            "prediction": prediction
        })

    accuracy = (correct / len(test_dataset)) * 100  # 정확도를 퍼센트로 계산

    # WandB 로깅
    wandb.log({
        "test_results": wandb.Table(
            columns=["file", "ground_truth", "prediction"],
            data=[[res["file"], res["ground_truth"], res["prediction"]] for res in test_results]
        ),
        "test_accuracy": accuracy
    })

    print(f"Test Accuracy: {accuracy}%")

# 테스트 실행
evaluate_and_log_test_data(test_dataset)
