import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

# ==========================================
# 1. 参数与数据格式化
# ==========================================
MODEL_NAME = "google/byt5-small"
JOINT_CSV = "germanic_joint_ipa_dataset.csv"
OUTPUT_DIR = "./byt5_grapheme_phoneme_reconstructor"

df_joint = pd.read_csv(JOINT_CSV)

# 核心：将文字与音标融合拼接
# 格式：English: free [/friː/] | German: frei [/fraɪ/] -> Target: frēo [/freːo/]
def format_joint_input(row):
    parts = []
    langs = [
        ("English", "Modern_English", "en_ipa"),
        ("German", "Modern_German", "de_ipa"),
        ("Dutch", "Modern_Dutch", "nl_ipa"),
        ("Swedish", "Modern_Swedish", "sv_ipa")
    ]
    for lang_name, word_col, ipa_col in langs:
        word = row[word_col]
        ipa = row[ipa_col]
        if str(word) != "-":
            parts.append(f"{lang_name}: {word} [{ipa}]")
    return " | ".join(parts)

df_joint["input_text"] = df_joint.apply(format_joint_input, axis=1)
# 目标端同样采用联合形式：单词 + 音标
df_joint["target_text"] = df_joint.apply(lambda r: f"{r['Target_Old_English']} [{r['oe_ipa']}]", axis=1)

# 划分数据集 (15% 验证集)
raw_dataset = Dataset.from_pandas(df_joint[["input_text", "target_text"]])
dataset_split = raw_dataset.train_test_split(test_size=0.15, seed=42)

print("联合输入样例:", dataset_split["train"][0]["input_text"])
print("联合目标样例:", dataset_split["train"][0]["target_text"])

# ==========================================
# 2. 模型与预处理
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=256, truncation=True)
    labels = tokenizer(text_target=examples["target_text"], max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = dataset_split["train"].map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
tokenized_val = dataset_split["test"].map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ==========================================
# 3. 严格限制空间的训练参数
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",         
    save_strategy="epoch",          # 每轮评估并尝试保存
    learning_rate=2e-4,            
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    
    # 核心空间限制配置：
    save_total_limit=1,             # 磁盘上最多保留 1 个 checkpoint
    load_best_model_at_end=True,   # 训练结束自动加载历史最佳模型
    metric_for_best_model="loss",   # 评估最佳模型的指标为 Loss
    
    num_train_epochs=12,            # 联合模型信息量大，12 轮足够
    predict_with_generate=True,
    fp16=False,                     # 使用 FP32 确保 ByT5 在 T4 上绝对稳定
    logging_steps=50,
    report_to="none"
)

# ==========================================
# 4. 开始训练
# ==========================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("\n--- 开始训练【音形联合重构模型】 ---")
trainer.train()

# 保存最终最佳的联合模型
trainer.save_model(os.path.join(OUTPUT_DIR, "best_joint_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_joint_model"))
print("训练完成！最佳联合模型已保存，临时 checkpoint 已自动释放。")