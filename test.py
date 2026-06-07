import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from phonemizer import phonemize
import sys

# ==========================================
# 1. 环境与联合模型加载
# ==========================================
MODEL_PATH = "./byt5_grapheme_phoneme_reconstructor/best_joint_model"

print("正在加载【音形联合重构模型】...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"\n❌ 加载模型失败！请确保你已经完成了联合训练，并且模型保存在 '{MODEL_PATH}'。")
    print(f"错误信息: {e}")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"✅ 模型加载成功！当前运行设备: {device}\n")

# ==========================================
# 2. 现代语言实时 IPA 转录工具 (G2P)
# ==========================================
def get_single_ipa(word, lang):
    if not word or word == "-":
        return "-"
    
    espeak_langs = {
        "en": "en-us",
        "de": "de",
        "nl": "nl",
        "sv": "sv"
    }
    
    try:
        # 调用本地的 espeak-ng 后端实时转录单个单词的音标
        phoneme = phonemize(
            [word],
            language=espeak_langs[lang],
            backend='espeak',
            strip=True
        )[0].strip()
        return f"/{phoneme}/" if phoneme else "-"
    except Exception as e:
        print(f"⚠️ 警告：无法转录 {lang} 单词 '{word}' 的音标 (请确保系统已安装 espeak-ng)")
        return "-"

# ==========================================
# 3. 联合预测核心逻辑
# ==========================================
def predict_joint_old_english(english="-", german="-", dutch="-", swedish="-"):
    parts = []
    langs = [
        ("English", english, "en"),
        ("German", german, "de"),
        ("Dutch", dutch, "nl"),
        ("Swedish", swedish, "sv")
    ]
    
    # 自动生成音标并进行格式化拼接
    for lang_name, word, lang_code in langs:
        if word != "-":
            # 核心：自动查找并生成 IPA 物理音标
            ipa = get_single_ipa(word, lang_code)
            parts.append(f"{lang_name}: {word} [{ipa}]")
            
    input_text = " | ".join(parts)
    print(f"\n🔍 格式化后的模型输入 (Input Text): \033[1;34m{input_text}\033[0m")
    
    # 将文本转换为 token 传入模型
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64, # 音形联合输出略长，增加最大长度限制
            num_beams=5,   # 使用 Beam Search
            early_stopping=True
        )
        
    # 解码得到答案 (因为训练目标是 word [ipa], 所以这里也会自动带音标输出)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

# ==========================================
# 4. 交互式用户主循环
# ==========================================
print("=================================================================")
print("  欢迎使用【古英语音形联合智能重构系统】 (Joint Reconstructor)")
print("  说明：输入现代日耳曼语单词，系统会自动转录音标并预测古英语源头。")
print("  提示：若无对应同源词，直接按【回车】跳过。输入 'q' 可退出系统。")
print("=================================================================")

while True:
    print("\n--- 请输入现代日耳曼语同源词 ---")
    eng = input("1. 现代英语 (English) [直接回车跳过]: ").strip() or "-"
    if eng.lower() in ['q', 'exit']: 
        print("系统已退出。")
        break
        
    ger = input("2. 现代德语 (German)  [直接回车跳过]: ").strip() or "-"
    if ger.lower() in ['q', 'exit']: break
    
    dut = input("3. 现代荷兰语 (Dutch)   [直接回车跳过]: ").strip() or "-"
    if dut.lower() in ['q', 'exit']: break
    
    swe = input("4. 现代瑞典语 (Swedish) [直接回车跳过]: ").strip() or "-"
    if swe.lower() in ['q', 'exit']: break
    
    # 输入校验
    if eng == "-" and ger == "-" and dut == "-" and swe == "-":
        print("❌ 错误：你必须至少提供一个现代语言的同源词！")
        continue
        
    print("\n正在后台生成物理音标并进行历史重构，请稍候...")
    try:
        result = predict_joint_old_english(english=eng, german=ger, dutch=dut, swedish=swe)
        print(f"👉 AI 重构的【古英语形式 (单词 + IPA)】为: \033[1;32m{result}\033[0m")
    except Exception as e:
        print(f"发生错误: {e}")