import os
import json
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    pipeline,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from evaluate import load as load_metric
from tqdm import tqdm

# 1. 데이터 로드 & split & prompt 전처리 (schema + db_id + table_name 포함)
def load_datasets():
    full = load_dataset(
        "json",
        data_files="/workspace/SQL2NLFullFineTuning/sql2nl_healthcare_train_dataset.jsonl",
        split="train"
    ).shuffle(seed=42)
    train = full.select(range(3200))
    valid = full.select(range(3200, 3600))
    test  = full.select(range(3600, 4000))

    def format_prompt(ex):
        cols = ", ".join(f"{c['name']}: {c['description']}" for c in ex["schema"]["columns"])
        prompt = (
            "Instruction: SQL 쿼리를 자연어 질의로 변환하세요.\n"
            f"### DB ID: {ex['db_id']}\n"
            f"### Table: {ex['table_name']}\n"
            f"### SQL Query:\n{ex['input_query']}\n"
            f"### Schema Columns:\n{cols}\n"
            "### Natural Language:\n"
        )
        return {"text": prompt + ex["output_utterance"]}

    remove_cols = ["input_query", "output_utterance", "schema", "db_id", "table_name"]
    train = train.map(format_prompt, remove_columns=remove_cols)
    valid = valid.map(format_prompt, remove_columns=remove_cols)
    test  = test.map(format_prompt, remove_columns=remove_cols)

    return DatasetDict({"train": train, "validation": valid, "test": test})

dataset = load_datasets()

# 2. Tokenizer
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.padding_side = "right"
tokenizer.model_max_length = 512

# 3. QLoRA: 4-bit 양자화 설정
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# 4. LoRA 어댑터 설정
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

# 5. TrainingArguments
training_args = TrainingArguments(
    output_dir="./sql2nl_qwen3_qlora_schema",
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

# 6. formatting 함수 (text 그대로 반환)
def formatting_func(example):
    return example["text"]

# 7. SFTTrainer 초기화
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    formatting_func=formatting_func,
    processing_class=tokenizer,
)

# 8. 학습 시작
trainer.train()

# 9. 모델 저장
out_dir = "./sql2nl_qwen3_qlora_schema_final"
os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print(f"✅ QLoRA 모델 저장 완료: {out_dir}")

# 10. 평가(metric & inference)
bleu      = load_metric("bleu")
meteor    = load_metric("meteor")
rouge     = load_metric("rouge")
bertscore = load_metric("bertscore")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"temperature":0.2, "do_sample":True},
    max_new_tokens=150
)

refs, preds, results = [], [], []
for ex in tqdm(dataset["test"], desc="QLoRA Test Eval"):
    sql, true = ex["text"].split("### Natural Language:\n")
    sql += "### Natural Language:\n"
    pred = pipe(sql)[0]["generated_text"][len(sql):]
    preds.append(pred)
    refs.append([true])
    results.append({
        "sql_query": sql.split("### SQL Query:\n")[1].split("### Schema Columns:")[0].strip(),
        "true_output": true,
        "pred_output": pred
    })

# 저장
with open(os.path.join(out_dir, "test_predictions.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 11. 메트릭 계산 (mid 제거)
def compute_metrics(refs, preds):
    bleu_score = bleu.compute(predictions=preds, references=refs)["bleu"]
    meteor_score = meteor.compute(predictions=preds, references=refs)["meteor"]
    rouge_scores = rouge.compute(predictions=preds, references=refs)
    bert_scores = bertscore.compute(predictions=preds, references=refs, lang="ko")

    return {
        "BLEU": bleu_score,
        "METEOR": meteor_score,
        "ROUGE-2": float(rouge_scores["rouge2"]),
        "ROUGE-L": float(rouge_scores["rougeL"]),
        "BERTScore-P": float(np.mean(bert_scores["precision"])),
        "BERTScore-R": float(np.mean(bert_scores["recall"])),
        "BERTScore-F1": float(np.mean(bert_scores["f1"])),
    }

metrics = compute_metrics(refs, preds)
with open(os.path.join(out_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\n=== QLoRA 최종 평가 결과 ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
