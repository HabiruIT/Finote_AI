# inference_multi.py
# Load slot_model + intent_model, run inference on input text, assemble JSON output.
import os, json, re
from datetime import datetime, timedelta
import dateparser

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

# ========== CONFIG ==========
SLOT_MODEL_DIR = "./slot_model"
INTENT_MODEL_DIR = "./intent_model"
LABEL_LIST = [
    "O",
    "B-AMOUNT","I-AMOUNT",
    "B-TIME","I-TIME",
    "B-NOTE","I-NOTE"
]
label_to_id = {l:i for i,l in enumerate(LABEL_LIST)}
id_to_label = {i:l for l,i in label_to_id.items()}
# ===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(SLOT_MODEL_DIR, use_fast=False)
slot_model = AutoModelForTokenClassification.from_pretrained(SLOT_MODEL_DIR).to(device)
intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_DIR).to(device)

# load intent_map
intent_map = {}
try:
    with open(os.path.join(INTENT_MODEL_DIR, "intent_map.json"), "r", encoding="utf-8") as fr:
        intent_map = json.load(fr)
        # invert map: {model_label_idx: original_categoryID}
        inv_intent = {v:int(k) for k,v in intent_map.items()}
except Exception as e:
    print("Warning: cannot load intent_map.json:", e)
    inv_intent = {}

# -------------------------
# Helpers: same tokenization+alignment strategy as training
# -------------------------
def tokenize_words_flat(words):
    flat_subwords = []
    word_map = []  # for each subword index -> word index
    for wi, w in enumerate(words):
        subws = tokenizer.tokenize(w)
        if len(subws) == 0:
            subws = tokenizer.tokenize(w if w.strip() else "<unk>")
        for s in subws:
            flat_subwords.append(s)
            word_map.append(wi)
    token_ids = tokenizer.convert_tokens_to_ids(flat_subwords)
    return flat_subwords, token_ids, word_map

def build_inputs_from_words(words: list):
    flat_subwords, token_ids, word_map = tokenize_words_flat(words)
    input_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
    # create labels dummy length for model input alignment (we don't need labels now)
    # attention mask:
    attention_mask = [1] * len(input_ids)
    return flat_subwords, token_ids, word_map, input_ids, attention_mask

# -------------------------
# Postprocess token predictions -> word-level labels and spans
# -------------------------
def logits_to_word_labels(pred_ids: list, word_map: list, labels_map=id_to_label):
    word_labels = []
    # find first subword index of each word
    word_first_subword = {}
    for si, wi in enumerate(word_map):
        if wi not in word_first_subword:
            word_first_subword[wi] = si
    num_words = max(word_map) + 1 if len(word_map)>0 else 0
    for wi in range(num_words):
        si = word_first_subword[wi]
        labid = int(pred_ids[si])
        word_labels.append(labels_map[labid])
    return word_labels

# -------------------------
# Entity extraction from word labels & original tokens
# -------------------------
def extract_entities_from_word_labels(words: list, word_labels: list):
    ents = {"AMOUNT": [], "TIME": [], "NOTE": []}
    cur_ent = None
    cur_tokens = []
    for w, lab in zip(words, word_labels):
        if lab == "O":
            if cur_ent is not None:
                ents[cur_ent].append(" ".join(cur_tokens))
                cur_ent = None
                cur_tokens = []
            continue
        if lab.startswith("B-"):
            if cur_ent is not None:
                ents[cur_ent].append(" ".join(cur_tokens))
            cur_ent = lab.split("-",1)[1]
            cur_tokens = [w]
        elif lab.startswith("I-"):
            typ = lab.split("-",1)[1]
            if cur_ent == typ:
                cur_tokens.append(w)
            else:
                cur_ent = typ
                cur_tokens = [w]
    if cur_ent is not None:
        ents[cur_ent].append(" ".join(cur_tokens))
    return ents

# -------------------------
# Normalizers
# -------------------------
# _amount_re = re.compile(r"(\d+(?:[.,]\d+)?)(\s*(k|kđ|kđ|nghìn|ngàn|triệu|tr|đ|vnđ)?)", flags=re.IGNORECASE)

def normalize_amount(text_fragment: str):
    if not text_fragment or not isinstance(text_fragment, str):
        return None

    text = text_fragment.lower().strip()

    #ĐƠN VỊ TIỀN
    money_units = [
        'k', 'nghìn', 'ngàn', 'triệu', 'tỉ', 'tỷ',
        'đ', 'đồng', 'vnd', 'vnđ', '₫'
    ]

    physical_units = [
        'kg', 'km', 'm', 'l', 'lit', 'lít', 'cái', 'tuổi', 'độ',
        'phút', 'giờ', 'ngày', 'mm', 'cm'
    ]

    context_words = ['hết', 'mất', 'giá', 'trả', 'tổng', 'tiền', 'phải trả', 'chi', 'mua']

    number_pattern = r"\d+(?:[.,]?\d+)*"

    #Nếu đầu vào chỉ toàn là số return luôn
    if re.fullmatch(number_pattern, text):
        clean_num = text.replace(".", "").replace(",", "")
        try:
            if int(float(clean_num)) < 1000:
                return int(float(clean_num)) * 1000
            else:
                return int(float(clean_num))
        except:
            return None

    # ===== DUYỆT TẤT CẢ CÁC SỐ TRONG CHUỖI =====
    all_numbers = []  # chứa tất cả số bắt được
    for match in re.finditer(number_pattern, text):
        num_str = match.group()
        start, end = match.span()

        all_numbers.append(num_str)

        after = text[end:end+10].strip()
        before = text[max(0, start-20):start].strip()

        clean_num = num_str.replace(".", "").replace(",", "")
        try:
            num_val = float(clean_num)
        except:
            continue

        # ===== ➋ Nếu sau số có đơn vị tiền → xử lý đúng hệ số =====
        for unit in money_units:
            if re.match(rf"^\s*{unit}\b", after):
                if unit in ['k', 'nghìn', 'ngàn']:
                    return int(num_val * 1000)
                elif unit == 'triệu':
                    return int(num_val * 1_000_000)
                elif unit in ['tỉ', 'tỷ']:
                    return int(num_val * 1_000_000_000)
                else:
                    return int(num_val)

        # ===== ➌ Nếu sau số là đơn vị vật lý → bỏ qua =====
        skip = False
        for unit in physical_units:
            if re.match(rf"^\s*{unit}\b", after):
                skip = True
                break
        if skip:
            continue

        # ===== ➍ Ngữ cảnh chỉ về tiền ("hết 30000", "mua 200") =====
        for ctx in context_words:
            if ctx in before:
                return int(num_val)

        # ===== ➎ Nếu không đoán được nhưng có dạng tiền → fallback sau =====
        # bỏ qua để xử lý cuối

    # ===== ➏ Nếu không xác định được số nào là tiền → lấy số đầu tiên =====
    if len(all_numbers) > 0:
        try:
            first = all_numbers[0].replace(",", "").replace(".", "")
            return int(float(first))
        except:
            return None

    return None

def normalize_time(text_time: str, reference: datetime = None):
    if not text_time or not isinstance(text_time, str):
        return None

    text = text_time.lower().strip()
    now = reference or datetime.now()
    dt = now

    # ======== 1️⃣. MATCH "5 phút trước", "2 giờ sau", "nửa tiếng trước" =========
    relative_pattern = re.search(
        r"(?:(nửa)\s*)?(\d+)?\s*(phút|p|giờ|h|tiếng|ngày)\s*(rưỡi|trước|sau)?",
        text
    )

    if relative_pattern:
        half = relative_pattern.group(1)
        num = relative_pattern.group(2)
        unit = relative_pattern.group(3)
        suffix = relative_pattern.group(4)

        # Chỉ áp dụng khi có suffix "trước" hoặc "sau" hoặc có "nửa"/"rưỡi"
        if suffix in ["trước", "sau"] or half or suffix == "rưỡi":
            value = float(num) if num else 0
            if half:
                value = 0.5
            if suffix == "rưỡi":   # "1 tiếng rưỡi"
                value += 0.5

            # Xác định đơn vị
            delta = timedelta()
            if unit in ["phút", "p"]:
                delta = timedelta(minutes=value)
            elif unit in ["giờ", "h", "tiếng"]:
                delta = timedelta(hours=value)
            elif unit == "ngày":
                delta = timedelta(days=value)

            if suffix == "sau":
                dt = now + delta
            else:
                dt = now - delta

            return dt.strftime("%Y-%m-%d %H:%M:%S")

    # ======== 2️⃣. HÔM QUA, HÔM NAY, MAI, HÔM KIA - CẬP NHẬT NGÀY TRƯỚC =========
    if "hôm kia" in text:
        dt = now - timedelta(days=2)
    elif "hôm qua" in text:
        dt = now - timedelta(days=1)
    elif "hôm nay" in text or "nay" in text:
        dt = now
    elif "ngày mai" in text or "mai" in text:
        dt = now + timedelta(days=1)
    elif "ngày kia" in text:
        dt = now + timedelta(days=2)

    # ======== 2.5️⃣. "Hôm qua lúc 6:00", "Hôm nay lúc 14:30", "Mai lúc 8:00" =========
    # Pattern mở rộng: hỗ trợ "lúc", "vào", "vào lúc" + giờ:phút hoặc chỉ giờ
    luc_pattern = re.search(
        r"(?:lúc|vào lúc|vào)\s*(\d{1,2})\s*(?:[:h giờ]\s*(\d{1,2})?)?",
        text
    )
    if luc_pattern:
        hour = int(luc_pattern.group(1))
        minute = int(luc_pattern.group(2)) if luc_pattern.group(2) else 0
        # Đảm bảo giờ hợp lệ (0-23)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            dt = dt.replace(hour=hour, minute=minute, second=0)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    # ======== 3️⃣. Giờ nâng cao: 2h, 2h30, 2 giờ 15, 2h kém 15, gần 3h ========
    # Pattern: optional "gần", hour, optional minute/kém
    time_pattern = re.search(
        r"(?:(gần)\s*)?(\d{1,2})\s*(?:h|giờ)\s*(?:kém\s*(\d{1,2})|(\d{1,2}))?\s*(?:p|phút)?",
        text
    )

    if time_pattern:
        is_near = bool(time_pattern.group(1))
        hour = int(time_pattern.group(2))
        minute = 0

        if time_pattern.group(3):  # "2h kém 15"
            minute = 60 - int(time_pattern.group(3))
            hour -= 1
        elif time_pattern.group(4):  # "2h15"
            minute = int(time_pattern.group(4))

        if is_near:  # "gần 3h" → trừ 10 phút
            minute -= 10
            if minute < 0:
                minute += 60
                hour -= 1
                if hour < 0:
                    hour = 23

        # Đảm bảo giờ hợp lệ (0-23)
        if 0 <= hour <= 23:
            dt = dt.replace(hour=hour, minute=max(0, min(59, minute)), second=0)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    # ======== 4️⃣. SÁNG / TRƯA / CHIỀU / TỐI + Kết hợp với ngày đã xác định ========
    time_of_day_set = False
    if "sáng" in text:
        dt = dt.replace(hour=8, minute=0, second=0)
        time_of_day_set = True
    elif "trưa" in text:
        dt = dt.replace(hour=12, minute=0, second=0)
        time_of_day_set = True
    elif "chiều" in text:
        dt = dt.replace(hour=17, minute=0, second=0)
        time_of_day_set = True
    elif "tối" in text or "đêm" in text:
        dt = dt.replace(hour=21, minute=0, second=0)
        time_of_day_set = True
    
    if time_of_day_set:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    # ======== 5️⃣. Chỉ có giờ đơn giản: "6:00", "14:30", "6h", "6 giờ" =========
    simple_time_pattern = re.search(
        r"(\d{1,2})\s*[:h giờ]\s*(\d{1,2})?\s*(?:p|phút)?",
        text
    )
    if simple_time_pattern:
        hour = int(simple_time_pattern.group(1))
        minute = int(simple_time_pattern.group(2)) if simple_time_pattern.group(2) else 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            dt = dt.replace(hour=hour, minute=minute, second=0)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    # ======== 6️⃣. FALLBACK: dateparser (cực mạnh) ========
    settings = {"RELATIVE_BASE": now, "PREFER_DATES_FROM": "past"}
    try:
        dp = dateparser.parse(text_time, settings=settings, languages=["vi"])
        if dp:
            return dp.strftime("%Y-%m-%d %H:%M:%S")
    except:
        pass

    # ======== 7️⃣. fallback cuối cùng ========
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# -------------------------
# Main inference function
# -------------------------
def assemble_transaction(text: str, reference: datetime=None):
    # naive whitespace tokenization for words (should match annotation tokenization)
    words = text.strip().split()
    flat_subwords, token_ids_base, word_map = tokenize_words_flat(words)
    # prepare input ids with special tokens (same as training)
    input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_base)
    # build attention mask
    attention_mask = [1] * len(input_ids)

    # convert to tensors (must be batch dim)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    attn_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)

    # Run slot model
    slot_model.eval()
    with torch.no_grad():
        out = slot_model(input_ids=input_ids_tensor, attention_mask=attn_tensor)
        logits = out.logits.cpu().numpy()[0]

    num_sub = len(token_ids_base)
    logits_sub = logits[1:1+num_sub] if logits.shape[0] >= num_sub+2 else logits[0:num_sub]
    pred_ids = np.argmax(logits_sub, axis=-1).tolist()  # length == num_sub

    # aggregate to word-level labels (pick first subword label)
    word_labels = logits_to_word_labels(pred_ids, word_map)

    # extract entities
    ents = extract_entities_from_word_labels(words, word_labels)

    # normalize amount/time
    amount = None
    if len(ents.get("AMOUNT", [])) > 0:
        # prefer first AMOUNT
        amount_text = ents["AMOUNT"][0]
        print(amount_text)
        amount = normalize_amount(amount_text)

    tx_time = None
    if len(ents.get("TIME", [])) > 0:
        time_text = ents["TIME"][0]
        print(time_text)
        tx_time = normalize_time(time_text, reference=reference)

    # note: if CATEGORY exists use that as note fallback, else NOTE
    note = None
    if len(ents.get("NOTE", [])) > 0:
        note = ents["NOTE"][0]
        print(note)
    else:
        # fallback: use whole text minus amount/time tokens (simple)
        note = text

    # Run intent model to predict categoryID (if available)
    enc = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k,v in enc.items()}
    intent_model.eval()
    with torch.no_grad():
        out = intent_model(**enc)
        logits_int = out.logits.cpu().numpy()[0]
        pred_int = int(np.argmax(logits_int))
        # invert mapping if available
        categoryID = inv_intent.get(pred_int, None) if inv_intent else pred_int

    result = {
        "categoryID": int(categoryID) if categoryID is not None else None,
        "amount": amount,
        "note": note,
        "transactiontime": tx_time
    }

    conf = {"amount": 0.0, "time": 0.0, "category": 0.0}
    try:
        # compute softmax of logits_sub on axis -1
        from scipy.special import softmax
        probs = softmax(logits_sub, axis=-1)
        # if amount exists, find predicted positions for first AMOUNT occurrence
        if len(ents.get("AMOUNT", [])) > 0:
            # find subword index for first AMOUNT (word_map)
            amount_word = ents["AMOUNT"][0].split()[0]
            # naive: find first word index where word == amount_word
            wi = None
            for i,w in enumerate(words):
                if w == amount_word:
                    wi = i; break
            if wi is not None:
                # get first subword index for this word
                for si, widx in enumerate(word_map):
                    if widx == wi:
                        conf["amount"] = float(probs[si, np.argmax(logits_sub[si])])
                        break
        # time conf
        if len(ents.get("TIME", [])) > 0:
            time_word = ents["TIME"][0].split()[0]
            wi = None
            for i,w in enumerate(words):
                if w == time_word:
                    wi = i; break
            if wi is not None:
                for si, widx in enumerate(word_map):
                    if widx == wi:
                        conf["time"] = float(probs[si, np.argmax(logits_sub[si])])
                        break
        # category conf from intent softmax
        probs_int = softmax(logits_int, axis=0)
        conf["category"] = float(np.max(probs_int))
    except Exception:
        # fallback: leave zeros
        pass

    # follow-up policy
    follow_up = None
    if result["amount"] is None or conf["amount"] < 0.7:
        follow_up = "Bạn chi bao nhiêu tiền vậy?"
    elif result["transactiontime"] is None or conf["time"] < 0.6:
        follow_up = "Giao dịch xảy ra vào ngày/giờ nào?"
    elif result["categoryID"] is None or conf["category"] < 0.6:
        follow_up = "Chi tiêu này thuộc mục gì? (ví dụ: xăng, ăn uống, mua sắm...)"

    return result, conf, follow_up, ents

def process(text: str, reference: datetime = None):
    result, conf, follow_up, ents = assemble_transaction(text, reference)
    return result
