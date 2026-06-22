import json, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          set_seed)
from datasets import Dataset
import warnings; warnings.filterwarnings("ignore")

set_seed(42)
LABEL_MAP = {"analysis":0,"hot_take":1,"reaction":2}
ID2L = {v:k for k,v in LABEL_MAP.items()}
NUM = 3

df = pd.read_csv("data/takemeter_nba_labeled.csv")
df["label_id"] = df["label"].map(LABEL_MAP)
df = df.dropna(subset=["label_id"]); df["label_id"]=df["label_id"].astype(int)

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label_id"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"])
train_df=train_df.reset_index(drop=True); val_df=val_df.reset_index(drop=True); test_df=test_df.reset_index(drop=True)

MODEL="distilbert-base-uncased"
tok=AutoTokenizer.from_pretrained(MODEL)
def tokenize(e): return tok(e["text"], truncation=True, max_length=256)
def mk(d): return Dataset.from_pandas(d[["text","label_id"]].rename(columns={"label_id":"labels"})).map(tokenize, batched=True)
trd, vad, ted = mk(train_df), mk(val_df), mk(test_df)
coll=DataCollatorWithPadding(tokenizer=tok)

model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM, id2label=ID2L, label2id=LABEL_MAP)
def cm_(ep):
    lo,la=ep; return {"accuracy":accuracy_score(la, np.argmax(lo,axis=-1))}
args=TrainingArguments(output_dir="./takemeter-model", num_train_epochs=3,
    per_device_train_batch_size=16, per_device_eval_batch_size=32, learning_rate=2e-5,
    weight_decay=0.01, warmup_steps=50, eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="accuracy",
    logging_steps=10, report_to="none", seed=42)
tr=Trainer(model=model, args=args, train_dataset=trd, eval_dataset=vad, data_collator=coll, compute_metrics=cm_)
tr.train()

out=tr.predict(ted)
pred=np.argmax(out.predictions,axis=-1); true=out.label_ids
probs=torch.nn.functional.softmax(torch.tensor(out.predictions),dim=-1).numpy()
acc=accuracy_score(true,pred)
names=[ID2L[i] for i in range(NUM)]
print("ACC", round(acc,3))
print(classification_report(true,pred,target_names=names,zero_division=0))
print("CM (rows=true):\n", confusion_matrix(true,pred))

rows=[]
for i in range(len(test_df)):
    rows.append({"text":test_df.iloc[i]["text"], "true":ID2L[true[i]], "pred":ID2L[pred[i]],
                 "conf":round(float(probs[i][pred[i]]),3),
                 "p_analysis":round(float(probs[i][0]),3),
                 "p_hot_take":round(float(probs[i][1]),3),
                 "p_reaction":round(float(probs[i][2]),3),
                 "correct":bool(true[i]==pred[i])})
json.dump(rows, open("test_predictions.json","w"), indent=2)
print("\nWRONG PREDICTIONS:")
for r in rows:
    if not r["correct"]:
        print(f'[{r["true"]} -> {r["pred"]} conf {r["conf"]}] {r["text"][:180]}')
print("\nSaved test_predictions.json")
