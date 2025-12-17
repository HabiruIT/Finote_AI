import os
import json
import random
from typing import List, Dict, Tuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import numpy as np
from seqeval.metrics import classification_report, f1_score

# CONFIG
MODEL_NAME = "vinai/phobert-base"
TRAIN_FILE = "data/train.jsonl"
OUTPUT_SLOT_DIR = "./slot_model"
OUTPUT_INTENT_DIR = "./intent_model"

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 4
LR = 2e-5
SEED = 42

LABEL_LIST = [
    "O",
    "B-AMOUNT","I-AMOUNT",
    "B-TIME","I-TIME",
    "B-NOTE","I-NOTE"
]
label_to_id = {l:i for i,l in enumerate(LABEL_LIST)}
id_to_label = {i:l for l,i in label_to_id.items()}

# For intent labels: will be inferred from dataset file (distinct categoryID values)
# ============================

def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed()

print("Loading tokenizer:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)  # explicit non-fast


def tokenize_words_and_align_labels(words: List[str], word_labels: List[str]) -> Tuple[List[int], List[int], List[str]]:
    flat_subwords = []
    aligned = []

    for w, lab in zip(words, word_labels):
        subws = tokenizer.tokenize(w)
        if len(subws) == 0:
            subws = tokenizer.tokenize(w if w.strip() else "<unk>")
        flat_subwords.extend(subws)
        if lab == "O":
            for _ in subws:
                aligned.append(label_to_id["O"])
        else:
            typ = lab.split("-",1)[1]
            aligned.append(label_to_id[lab])
            for _ in subws[1:]:
                i_label = "I-" + typ
                aligned.append(label_to_id[i_label])
    token_ids = tokenizer.convert_tokens_to_ids(flat_subwords)
    return token_ids, aligned, flat_subwords

print("Loading dataset:", TRAIN_FILE)
raw = []
with open(TRAIN_FILE, 'r', encoding='utf-8') as fr:
    for line in fr:
        if not line.strip(): continue
        obj = json.loads(line)
        raw.append(obj)

random.shuffle(raw)
intent_set = sorted(list({int(x.get("categoryID", -1)) for x in raw}))
intent_map = {v:i for i,v in enumerate(intent_set)}
print("Intent categories found:", intent_map)

examples = []
for ex in raw:
    text = ex["text"]
    tokens = ex["tokens"]
    labels = ex["labels"]
    if len(tokens) != len(labels):
        raise ValueError(f"tokens/labels length mismatch for: {text}")
    token_ids, aligned_labels, flat_subwords = tokenize_words_and_align_labels(tokens, labels)
    input_ids_with_special = tokenizer.build_inputs_with_special_tokens(token_ids)

    labels_with_special = [-100] + aligned_labels + [-100]
    assert len(input_ids_with_special) == len(labels_with_special), f"Length mismatch after adding special tokens for: {text}"

    attention_mask = [1] * len(input_ids_with_special)

    cat_raw = int(ex.get("categoryID", -1))
    intent_label = intent_map[cat_raw]

    examples.append({
        "text": text,
        "input_ids": input_ids_with_special,
        "attention_mask": attention_mask,
        "labels": labels_with_special,
        "intent": intent_label,
        "tokens": tokens,
        "subtokens": flat_subwords
    })

split_idx = int(0.9 * len(examples)) if len(examples) > 10 else int(0.8 * len(examples))
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]
print(f"Total examples: {len(examples)}, train: {len(train_examples)}, val: {len(val_examples)}")

class SlotDataset(Dataset):
    def __init__(self, exs):
        self.exs = exs
    def __len__(self):
        return len(self.exs)
    def __getitem__(self, idx):
        e = self.exs[idx]
        return {
            "input_ids": torch.tensor(e["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(e["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(e["labels"], dtype=torch.long)
        }

def collate_batch(batch):
    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_padded = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attn_padded,
        "labels": labels_padded
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_slot_labels = len(LABEL_LIST)
slot_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_slot_labels)
slot_model.to(device)

num_intent_labels = len(intent_map)
intent_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_intent_labels)
intent_model.to(device)

train_loader = DataLoader(SlotDataset(train_examples), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(SlotDataset(val_examples), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

optim_slot = AdamW(slot_model.parameters(), lr=LR)

print("Start training slot model...")
for epoch in range(EPOCHS):
    slot_model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"slot train epoch {epoch+1}/{EPOCHS}"):
        optim_slot.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = slot_model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim_slot.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader) if len(train_loader)>0 else 0.0
    print(f"Epoch {epoch+1} slot avg loss: {avg_loss:.4f}")

    slot_model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            out = slot_model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits.cpu().numpy()
            pred_ids = np.argmax(logits, axis=-1)
            for i in range(pred_ids.shape[0]):
                pred_seq = []
                true_seq = []
                for j in range(pred_ids.shape[1]):
                    if labels[i,j] == -100:
                        continue
                    pred_seq.append(id_to_label[int(pred_ids[i,j])])
                    true_seq.append(id_to_label[int(labels[i,j])])
                preds_all.append(pred_seq)
                labels_all.append(true_seq)
    if len(labels_all) > 0:
        f1 = f1_score(labels_all, preds_all)
        print(f"Validation seq F1: {f1:.4f}")
    else:
        print("No validation labels to compute F1.")

os.makedirs(OUTPUT_SLOT_DIR, exist_ok=True)
slot_model.save_pretrained(OUTPUT_SLOT_DIR)
tokenizer.save_pretrained(OUTPUT_SLOT_DIR)
print("Saved slot model to", OUTPUT_SLOT_DIR)

class IntentDataset(Dataset):
    def __init__(self, exs):
        self.input_ids = []
        self.attn = []
        self.labels = []
        for e in exs:
            enc = tokenizer(e["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
            self.input_ids.append(torch.tensor(enc["input_ids"], dtype=torch.long))
            self.attn.append(torch.tensor(enc["attention_mask"], dtype=torch.long))
            self.labels.append(torch.tensor(e["intent"], dtype=torch.long))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attn[idx], "labels": self.labels[idx]}

train_intent_ds = IntentDataset(train_examples)
val_intent_ds = IntentDataset(val_examples)

train_loader_intent = DataLoader(train_intent_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader_intent = DataLoader(val_intent_ds, batch_size=BATCH_SIZE, shuffle=False)

optim_intent = AdamW(intent_model.parameters(), lr=LR)

print("Start training intent model...")
for epoch in range(EPOCHS):
    intent_model.train()
    tot_loss = 0.0
    for batch in tqdm(train_loader_intent, desc=f"intent train epoch {epoch+1}/{EPOCHS}"):
        optim_intent.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = intent_model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim_intent.step()
        tot_loss += loss.item()
    print(f"Epoch {epoch+1} intent avg loss: {tot_loss/len(train_loader_intent):.4f}")

    intent_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader_intent:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            out = intent_model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits.cpu().numpy()
            preds = np.argmax(logits, axis=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    if len(all_labels) > 0:
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        print(f"Validation intent accuracy: {acc:.4f}")

# save intent model + tokenizer
os.makedirs(OUTPUT_INTENT_DIR, exist_ok=True)
intent_model.save_pretrained(OUTPUT_INTENT_DIR)
tokenizer.save_pretrained(OUTPUT_INTENT_DIR)
print("Saved intent model to", OUTPUT_INTENT_DIR)

# also dump intent_map to disk
with open(os.path.join(OUTPUT_INTENT_DIR, "intent_map.json"), "w", encoding="utf-8") as fw:
    json.dump(intent_map, fw, ensure_ascii=False, indent=2)

print("Training finished.")