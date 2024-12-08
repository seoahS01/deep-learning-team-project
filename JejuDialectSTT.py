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
import torch

wandb.init(project="jejudialectstt")

torch.cuda.empty_cache()

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available, using CPU.")

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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

    # # 데이터셋 크기를 제한 -> 110개만 학습 돌려보기
    # train_dataset = train_dataset.select(range(100))  # 첫 100개의 샘플만 사용
    # valid_dataset = valid_dataset.select(range(10))  # 첫 10개의 샘플만 사용

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

        # 디버깅: 라벨 출력
        print("Decoded label:", processor.tokenizer.decode(labels[0].tolist(), skip_special_tokens=True))
        print("Input text:", batch["text"])
        print("Labels shape:", labels.shape)

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
    per_device_train_batch_size=4,  # GPU 메모리 문제를 피하기 위해 배치 크기 감소
    evaluation_strategy="epoch",
    num_train_epochs=10,
    fp16=False,  # GPU 성능 최적화를 위해 FP16 사용
    max_grad_norm=1.0,
    save_steps=1000,  # GPU 메모리 및 저장 공간을 고려하여 저장 주기 증가
    eval_steps=1000,  # 평가 주기 증가
    logging_steps=200,  # 로깅 주기 증가
    learning_rate=1e-5,
    warmup_steps=500,
    save_total_limit=2,
    gradient_accumulation_steps=2,  # 배치 크기를 줄였으므로 누적 스텝 증가
    dataloader_num_workers=2,  # 데이터 로드 속도 조절을 위해 워커 수 감소
    report_to="wandb",
    gradient_checkpointing=False  # 메모리 절약을 위해 gradient checkpointing 활성화
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

# Step 9: 트레인 데이터 중 하나로 예측 확인
train_sample = train_dataset[0]  # 트레인 데이터 중 하나를 가져옵니다.
input_values = torch.tensor(train_sample["input_values"]).unsqueeze(0).to("cuda")  # 입력값 변환 및 GPU로 이동

with torch.no_grad():
    logits = model(input_values).logits  # 모델로부터 logits 출력
    print("Logits shape:", logits.shape)  # 디버깅: logits의 차원 출력
    print("Logits:", logits)  # 디버깅: logits 출력
    predicted_ids = torch.argmax(logits, dim=-1)  # 가장 높은 확률의 ID 추출
    print("Predicted IDs:", predicted_ids)  # 디버깅: 예측된 ID 출력

# 라벨 디코딩
ground_truth_decoded = processor.batch_decode([train_sample["labels"]], skip_special_tokens=True)[0]
print("Decoded Ground Truth:", ground_truth_decoded)  # 디버깅: 디코딩된 라벨 출력

# 예측값 디코딩
transcription = processor.batch_decode(predicted_ids)[0]
print("Prediction:", transcription)  # 디코딩된 예측 출력

# 오디오 파일 로드
audio_input, sampling_rate = sf.read("/home/aix23606/seoah/JejuDialectSTT/data_sample/Trimmed_data_2인발화/talk_set1_collectorjj14_speakerjj59_speakerjj60_4_0_121order_1.0.wav")
# 샘플링 레이트 확인
if sampling_rate != 16000:
    raise ValueError("모델은 16kHz 샘플링 레이트를 기대합니다. 오디오 파일을 리샘플링하세요.")
# 오디오 전처리
input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True).input_values
# 모델 추론
with torch.no_grad():
    logits = model(input_values).logits
# 예측된 토큰 ID를 텍스트로 디코딩
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print("Transcription:", transcription[0])