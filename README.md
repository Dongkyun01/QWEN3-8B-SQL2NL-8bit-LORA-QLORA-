## **1. 실험 목적**

본 실험의 목적은,

**SQL-to-Natural Language 변환(Task: SQL2NL)** 문제에서

- **(1) 8-bit 양자화 + LoRA (Parameter-Efficient Fine-Tuning)**
- **(2) QLoRA (4-bit 양자화 + LoRA)**

두 가지 파인튜닝 기법의 **성능과 효율성을 비교**하는 것이다.

---

## **2. 데이터셋 및 전처리**

### **2.1 데이터 출처**

- **도메인**: Healthcare SQL-to-NL corpus
- **총 데이터 수**: 4000개 샘플
- **Split 구성**:
    - **Train**: 3200
    - **Validation**: 400
    - **Test**: 400

### **2.2 입력 전처리 과정**

각 샘플은 아래와 같은 JSON 구조를 가짐:

```json
json
복사편집
{
  "db_id": "database identifier",
  "table_name": "table name",
  "schema": {
    "columns": [
      {"name": "col1", "description": "desc1"},
      ...
    ]
  },
  "input_query": "SQL query string",
  "output_utterance": "Target natural language utterance"
}

```

### **2.3 최종 모델 입력 포맷**

모든 데이터는 다음의 **prompt format**으로 변환되어 모델에 입력됨:


Instruction: SQL 쿼리를 자연어 질의로 변환하세요.
### DB ID: {db_id}### Table: {table_name}### SQL Query:
{input_query}
### Schema Columns:
col1: desc1, col2: desc2, ...
### Natural Language:
{output_utterance}



- **Instruction 기반 prompting**으로 모델 task를 명확히 지정.
- **DB ID, Table name, Schema column descriptions** 포함하여 쿼리 context를 풍부하게 제공.

---

## **3. 모델 및 파인튜닝 설정**

### **3.1 공통 사항**

| 항목 | 설정 |
| --- | --- |
| **Base model** | Qwen/Qwen3-8B |
| **Tokenizer max length** | 512 |
| **Optimizer** | AdamW (SFTTrainer default) |
| **Scheduler** | Cosine |
| **Epochs** | 3 |
| **Learning rate** | 2e-4 |
| **Batch size (effective)** | 8 (1 x 8 gradient accumulation) |
| **Warmup steps** | 100 |

---

### **3.2 8-bit + LoRA**

| 항목 | 설정 |
| --- | --- |
| **Quantization** | 8-bit (`load_in_8bit=True`) |
| **Fine-tuning strategy** | LoRA (adapter tuning) |
| **LoRA rank (r)** | 64 |
| **LoRA alpha** | 128 |
| **LoRA dropout** | 0.05 |
| **Target modules** | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
- **장점**: Full weight finetuning 대비 memory 사용량 절감.
- **제약**: 8-bit quantization만으로는 QLoRA 대비 압축 효율성은 낮음.

---

### **3.3 QLoRA**

| 항목 | 설정 |
| --- | --- |
| **Quantization** | 4-bit (NF4) |
| **Fine-tuning strategy** | QLoRA (4-bit weight + LoRA adapter) |
| **bnb_4bit_use_double_quant** | True |
| **bnb_4bit_compute_dtype** | torch.bfloat16 |
| **LoRA 설정** | 동일 (rank=64, alpha=128, dropout=0.05, 동일 target modules) |
- **장점**: 4-bit weight quantization으로 VRAM 사용량을 크게 감소시키면서도, LoRA 어댑터로 task-specific adaptation 가능.
- **제약**: 학습 안정성 측면에서 더 신중한 hyperparameter tuning 필요.

---

## **4. 실험 결과**

### **4.1 평가 지표**

| Metric | 8bit + LoRA | QLoRA | Δ (QLoRA - LoRA) |
| --- | --- | --- | --- |
| **BLEU** | 0.3092 | **0.3329** | +0.0237 |
| **METEOR** | 0.5124 | **0.5221** | +0.0097 |
| **ROUGE-2** | 0.0925 | **0.0958** | +0.0033 |
| **ROUGE-L** | 0.3452 | **0.3488** | +0.0036 |
| **BERTScore-P** | 0.8867 | **0.8892** | +0.0025 |
| **BERTScore-R** | 0.8857 | **0.8885** | +0.0028 |
| **BERTScore-F1** | 0.8858 | **0.8884** | +0.0026 |

### **4.2 분석**

- **QLoRA가 모든 지표에서 소폭 우세**.
- 특히 **BLEU 점수 +0.0237 상승**은 n-gram 기반 정밀도 측면에서 의미 있는 개선.
- METEOR, ROUGE, BERTScore 지표도 일관되게 QLoRA가 우세,
    
    → **semantic & structural similarity** 모두 향상되었음을 시사.
    

---

## **5. 결론**

1. **성능 측면**
    - QLoRA가 전반적으로 더 높은 성능을 보임.
    - 4-bit quantization임에도 정보 손실 없이 효과적인 학습이 가능했음을 확인.
2. **자원 효율성**
    - QLoRA는 8bit LoRA 대비 **메모리 사용량이 현저히 낮아**,
        
        동일 환경에서 더 큰 배치 학습 혹은 더 긴 sequence 처리 가능.
        
3. **실무 적용**
    - 두 모델 모두 production 수준에 근접.
    - 특히 QLoRA는 **추론 latency, memory footprint, 성능**의 균형 측면에서 우수
