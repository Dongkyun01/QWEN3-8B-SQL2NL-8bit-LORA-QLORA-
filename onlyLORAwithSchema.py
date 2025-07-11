import os
import json
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from evaluate import load as load_metric
from tqdm import tqdm
import numpy as np

# 1. 데이터 로드 및 schema+db_id+table_name 포함 전처리
def load_datasets():
    full = load_dataset(
        "json",
        data_files="/workspace/SQL2NLFullFineTuning/sql2nl_healthcare_train_dataset.jsonl",
        split="train"
    ).shuffle(seed=42)

    train = full.select(range(3200))
    valid = full.select(range(3200, 3600))
    test  = full.select(range(3600, 4000))

    def format_prompt(example):
        # 컬럼 설명
        cols = ", ".join(
            f"{c['name']}: {c['description']}"
            for c in example["schema"]["columns"]
        )
        prompt = (
            "Instruction: SQL 쿼리를 자연어 질의로 변환하세요.\n"
            f"### DB ID: {example['db_id']}\n"
            f"### Table: {example['table_name']}\n"
            f"### SQL Query:\n{example['input_query']}\n"
            f"### Schema Columns:\n{cols}\n"
            "### Natural Language:\n"
        )
        return {"text": prompt + example["output_utterance"]}

    remove_cols = ["input_query", "output_utterance", "schema", "db_id", "table_name"]
    train = train.map(format_prompt, remove_columns=remove_cols)
    valid = valid.map(format_prompt, remove_columns=remove_cols)
    test  = test.map(format_prompt, remove_columns=remove_cols)

    return DatasetDict({"train": train, "validation": valid, "test": test})

dataset = load_datasets()

# 2. Tokenizer & Model 준비
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.padding_side = "right"
tokenizer.model_max_length = 512

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True   # 8-bit quantization으로 VRAM 절감
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA 설정
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj","v_proj","k_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 4. 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./sql2nl_qwen3_lora_schema_db",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    bf16=True,
    report_to="none",
    ddp_find_unused_parameters=False,
    save_total_limit=3,
    load_best_model_at_end=True,
)

# 5. formatting 함수 (문자열 그대로 반환)
def formatting_func(example):
    return example["text"]

# 6. Trainer 초기화
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    formatting_func=formatting_func,
    processing_class=tokenizer,
)

# 7. 학습 시작
trainer.train()

# 8. 모델 저장
output_dir = "./sql2nl_qwen3_lora_schema_db_final"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ 모델 저장 완료: {output_dir}")

# 9. 평가 준비
bleu      = load_metric("bleu")
meteor    = load_metric("meteor")
rouge     = load_metric("rouge")
bertscore = load_metric("bertscore")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"temperature":0.2,"do_sample":True},
    max_new_tokens=150
)

# 10. 테스트셋 평가
references, predictions, results = [], [], []
for ex in tqdm(dataset["test"], desc="LoRA-DB/Table Test Eval"):
    sql, truth = ex["text"].split("### Natural Language:\n")
    sql += "### Natural Language:\n"
    pred = pipe(sql)[0]["generated_text"][len(sql):]
    predictions.append(pred)
    references.append([truth])
    results.append({
        "sql_query": sql.split("### SQL Query:\n")[1].split("### Schema Columns:")[0].strip(),
        "true_output": truth,
        "pred_output": pred
    })

# 11. 결과 저장
with open(os.path.join(output_dir, "test_preds.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 12. 지표 계산
def compute_metrics(refs, preds):
    b = bleu.compute(predictions=preds, references=refs)["bleu"]
    m = meteor.compute(predictions=preds, references=refs)["meteor"]
    r2 = rouge.compute(predictions=preds, references=refs)["rouge2"]
    rl = rouge.compute(predictions=preds, references=refs)["rougeL"]
    bs = bertscore.compute(predictions=preds, references=refs, lang="ko")
    return {
        "BLEU": b,
        "METEOR": m,
        "ROUGE-2": r2,
        "ROUGE-L": rl,
        "BERTScore-P": np.mean(bs["precision"]),
        "BERTScore-R": np.mean(bs["recall"]),
        "BERTScore-F1": np.mean(bs["f1"]),
    }

metrics = compute_metrics(references, predictions)
with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\n=== 최종 평가 결과 ===")
for k,v in metrics.items():
    print(f"{k}: {v:.4f}")
