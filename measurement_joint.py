import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from phonemizer import phonemize
from tqdm import tqdm

# ==========================================
# 1. 基础配置
# ==========================================
MODEL_PATH = "./byt5_grapheme_phoneme_reconstructor/best_joint_model"
DEV_CSV = "actual_validation_set.csv"  # 确保路径一致
OUTPUT_CSV = "joint_dev_evaluation_results.csv"

# ==========================================
# 2. 莱文斯坦编辑距离计算器 (已修正缩进 Bug)
# ==========================================
def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1] # ✅ 修正：已经移出外层 for 循环！

# ==========================================
# 3. 现代语言实时 G2P 转换器
# ==========================================
ipa_cache = {}
def get_single_ipa(word, lang):
    if not word or word == "-":
        return "-"
    cache_key = (word, lang)
    if cache_key in ipa_cache:
        return ipa_cache[cache_key]
        
    espeak_langs = {
        "en": "en-us", "de": "de", "nl": "nl", "sv": "sv"
    }
    try:
        phoneme = phonemize(
            [word],
            language=espeak_langs[lang],
            backend='espeak',
            strip=True
        )[0].strip()
        result = f"/{phoneme}/" if phoneme else "-"
        ipa_cache[cache_key] = result
        return result
    except Exception as e:
        return "-"

# ==========================================
# 4. 古英语历史音标生成器
# ==========================================
def oe_to_ipa(word):
    if pd.isna(word) or word == "-":
        return "-"
    word = str(word).lower().strip().replace("*", "")
    vowel_map = {
        'ā': 'ɑː', 'ē': 'eː', 'ī': 'iː', 'ō': 'oː', 'ū': 'uː', 'ȳ': 'yː',
        'ǣ': 'æː', 'æ': 'æ', 'y': 'y', 'a': 'ɑ', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u'
    }
    word = word.replace("ēo", "eːo").replace("eo", "eo").replace("ēa", "æːɑ").replace("ea", "æɑ")
    word = word.replace("sc", "ʃ").replace("cg", "dʒ")
    word = re.sub(r'c([eiyæǣ])', r'tʃ\1', word)
    word = re.sub(r'g([eiyæǣ])', r'j\1', word)
    v_pattern = r'[aeiouāēīōūæy]'
    word = re.sub(f'({v_pattern})f({v_pattern})', r'\1v\2', word)
    word = re.sub(f'({v_pattern})s({v_pattern})', r'\1z\2', word)
    word = re.sub(f'({v_pattern})[þð]({v_pattern})', r'\1ð\2', word)
    word = word.replace("þ", "θ").replace("ð", "θ").replace("c", "k")
    for oe_v, ipa_v in vowel_map.items():
        word = word.replace(oe_v, ipa_v)
    return f"/{word}/"

# ==========================================
# 5. 模型加载与评估流程
# ==========================================
print("正在载入音形联合模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"载入 Dev 集: '{DEV_CSV}'...")
df_dev = pd.read_csv(DEV_CSV)

results = []
perfect_word_matches = 0
total_word_edit_dist = 0

# 统计各编辑距离区间的分布
dist_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, "4+": 0}

for idx, row in tqdm(df_dev.iterrows(), total=len(df_dev)):
    raw_input = str(row["input_text"])
    target_word = str(row["target_text"]).strip()
    
    # 5.1 解析并构建音形联合输入格式
    parts = []
    matches = re.findall(r"(\w+):\s*([^\|]+)", raw_input)
    for lang_name, word_val in matches:
        word_val = word_val.strip()
        lang_map = {"English": "en", "German": "de", "Dutch": "nl", "Swedish": "sv"}
        lang_code = lang_map.get(lang_name)
        if lang_code and word_val != "-":
            ipa = get_single_ipa(word_val, lang_code)
            parts.append(f"{lang_name}: {word_val} [{ipa}]")
            
    joint_input = " | ".join(parts)
    
    # 5.2 模型预测
    inputs = tokenizer(joint_input, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=5, early_stopping=True)
    pred_joint = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # 5.3 拆分模型预测的拼写和音标
    pred_word, pred_ipa = pred_joint, "-"
    if " [" in pred_joint:
        split_parts = pred_joint.split(" [")
        pred_word = split_parts[0].strip()
        pred_ipa = split_parts[1].replace("]", "").strip()
        
    # 5.4 生成目标的标准音标
    target_ipa = oe_to_ipa(target_word)
    
    # 5.5 计算编辑距离
    word_dist = edit_distance(pred_word, target_word)
    ipa_dist = edit_distance(pred_ipa, target_ipa) if pred_ipa != "-" and target_ipa != "-" else -1
    
    total_word_edit_dist += word_dist
    if word_dist == 0:
        perfect_word_matches += 1
        
    # 区间统计
    if word_dist <= 4:
        dist_counts[word_dist] += 1
    else:
        dist_counts["4+"] += 1
        
    results.append({
        "original_input": raw_input,
        "joint_input": joint_input,
        "target_word": target_word,
        "predicted_word": pred_word,
        "predicted_ipa": pred_ipa,
        "word_edit_distance": word_dist,
        "ipa_edit_distance": ipa_dist
    })

# ==========================================
# 6. 生成并打印评估报告 (对应 Ab Antiquo 论文 Table 1 格式)
# ==========================================
total_samples = len(df_dev)
accuracy = (perfect_word_matches / total_samples) * 100
avg_word_dist = total_word_edit_dist / total_samples

acc_0 = accuracy
acc_1 = (sum(dist_counts[i] for i in range(2)) / total_samples) * 100
acc_2 = (sum(dist_counts[i] for i in range(3)) / total_samples) * 100
acc_3 = (sum(dist_counts[i] for i in range(4)) / total_samples) * 100
acc_4 = (sum(dist_counts[i] for i in range(5)) / total_samples) * 100

print("\n" + "="*50)
print("             音形联合模型 Dev 集评估报告")
print("="*50)
print(f"评估样本总数: {total_samples}")
print(f"平均拼写编辑距离 (Avg Word Edit Distance): {avg_word_dist:.4f}")
print("-"*50)
print("编辑距离分布累加率 (对应论文 Table 1 格式):")
print(f"  距离 = 0 (完全完美重构): {acc_0:.2f}%")
print(f"  距离 <= 1             : {acc_1:.2f}%")
print(f"  距离 <= 2             : {acc_2:.2f}%")
print(f"  距离 <= 3             : {acc_3:.2f}%")
print(f"  距离 <= 4             : {acc_4:.2f}%")
print("="*50)

# 保存详细对齐结果
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"评估完成！详细对齐和对比数据已保存至：'{OUTPUT_CSV}'")