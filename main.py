

# Step 1: 환경 설정

# 필요한 라이브러리 설치
!pip install transformers
!pip install datasets
!pip install torch
!pip install google-cloud-storage

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from google.cloud import storage

# Step 2: 구클에서 데이터 준비

# 구클 클라이언트 사전설정
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """GCS에서 파일을 다운로드합니다."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")

# 구클에서 데이터 다운로드
bucket_name = "your-bucket-name"
source_blob_name = "path/to/fake_info.txt"
destination_file_name = "fake_info.txt"
download_from_gcs(bucket_name, source_blob_name, destination_file_name)

# 데이터를 로드
data_files = {"train": destination_file_name}
dataset = load_dataset('text', data_files=data_files)

# Step 3: 모델 로드 및 파인튜닝

# Llama3-8B 모델 로드 (모델 이름은 예시입니다. 실제 사용 시 정확한 모델 이름을 사용하세요)
model_name = "facebook/llama3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 데이터 전처리
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# 모델 파인튜닝
trainer.train()



# 모델 저장
model.save_pretrained("./fine_tuned_llama3_8b")
tokenizer.save_pretrained("./fine_tuned_llama3_8b")


''' def generate_fake_info(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)'''


def chat_with_model():
    print("허위 정보 챗봇에 오신 것을 환영합니다. '종료'라고 입력하면 대화를 종료합니다.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "종료":
            print("Chatbot: 대화를 종료합니다.")
            break
        response = generate_fake_info(prompt)
        print(f"Chatbot: {response}")


chat_with_model()
