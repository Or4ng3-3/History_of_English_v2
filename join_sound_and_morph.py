import pandas as pd
from phonemizer import phonemize
import re

CSV_FILE = "germanic_parallel_dataset.csv"
OUTPUT_JOINT_CSV = "germanic_joint_ipa_dataset.csv"

# ==========================================
# 1. 历史古英语 (Old English) 规则音标转换器
# ==========================================
def oe_to_ipa(word):
    if pd.isna(word) or word == "-":
        return "-"
    word = str(word).lower().strip().replace("*", "")
    
    # 元音及长音符映射
    vowel_map = {
        'ā': 'ɑː', 'ē': 'eː', 'ī': 'iː', 'ō': 'oː', 'ū': 'uː', 'ȳ': 'yː',
        'ǣ': 'æː', 'æ': 'æ', 'y': 'y', 'a': 'ɑ', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u'
    }
    
    # 双元音处理
    word = word.replace("ēo", "eːo").replace("eo", "eo")
    word = word.replace("ēa", "æːɑ").replace("ea", "æɑ")
    
    # 辅音组合
    word = word.replace("sc", "ʃ")
    word = word.replace("cg", "dʒ")
    
    # 腭化音近似处理 (c, g 在前元音前腭化)
    word = re.sub(r'c([eiyæǣ])', r'tʃ\1', word)
    word = re.sub(r'g([eiyæǣ])', r'j\1', word)
    
    # 剩余辅音回退
    word = word.replace("c", "k")
    # 介于元音之间的擦音浊化 (f->v, s->z, þ/ð->ð)
    v_pattern = r'[aeiouāēīōūæy]'
    word = re.sub(f'({v_pattern})f({v_pattern})', r'\1v\2', word)
    word = re.sub(f'({v_pattern})s({v_pattern})', r'\1z\2', word)
    word = re.sub(f'({v_pattern})[þð]({v_pattern})', r'\1ð\2', word)
    
    # 剩余默认辅音
    word = word.replace("þ", "θ").replace("ð", "θ")
    
    # 映射剩余单元音
    for oe_v, ipa_v in vowel_map.items():
        word = word.replace(oe_v, ipa_v)
        
    return f"/{word}/"

# ==========================================
# 2. 现代语言批量 IPA 转录器（去重防缩水修正版）
# ==========================================
def get_ipa_bulk(words_list, lang_code):
    # 将输入转为标准的 string 列表，并过滤掉空值和 "-"
    words_list_str = [str(w).strip() if pd.notna(w) else "-" for w in words_list]
    
    # 提取所有不为 "-" 的唯一单词进行转录，避免重复计算，同时防止空字符串被 phonemizer 丢弃
    unique_words = list(set([w for w in words_list_str if w != "-"]))
    
    espeak_langs = {
        "en": "en-us",
        "de": "de",
        "nl": "nl",
        "sv": "sv"
    }
    
    word_to_ipa = {}
    if unique_words:
        try:
            # 仅转录去重后的有效单词
            phonemes = phonemize(
                unique_words,
                language=espeak_langs[lang_code],
                backend='espeak',
                strip=True,
                njobs=4
            )
            # 建立 单词 -> 音标 的字典映射
            for word, ph in zip(unique_words, phonemes):
                word_to_ipa[word] = f"/{ph.strip()}/" if ph.strip() else "-"
        except Exception as e:
            print(f"转录语言 {lang_code} 失败: {e}")
            
    # 根据映射字典，1:1 还原出与原始输入完全等长的列表
    result = []
    for w in words_list_str:
        if w == "-":
            result.append("-")
        else:
            result.append(word_to_ipa.get(w, "-"))
            
    return result
# ==========================================
# 3. 运行清洗与构建
# ==========================================
print("正在载入原始平行词表...")
df = pd.read_csv(CSV_FILE)

print("正在生成现代语言的物理音标 (IPA)...")
df["en_ipa"] = get_ipa_bulk(df["Modern_English"].tolist(), "en")
df["de_ipa"] = get_ipa_bulk(df["Modern_German"].tolist(), "de")
df["nl_ipa"] = get_ipa_bulk(df["Modern_Dutch"].tolist(), "nl")
df["sv_ipa"] = get_ipa_bulk(df["Modern_Swedish"].tolist(), "sv")

print("正在根据历史语音学规则生成古英语音标...")
df["oe_ipa"] = df["Target_Old_English"].apply(oe_to_ipa)

# 保存音标对齐结果
df.to_csv(OUTPUT_JOINT_CSV, index=False, encoding="utf-8")
print(f"🎉 成功！音形联合数据集已保存至 '{OUTPUT_JOINT_CSV}'。")