#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import re
import os
import wandb
import string
import numpy as np
import unicodedata
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import flash_attn
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import sys
sys.path.append("../../src")
import util.preprocessing_util as util


# # Setup

# In[3]:


DATA_DIR = Path("../../data/dev/processed")
DATASET_NAME = "medical_data.csv"


# In[4]:


data = pd.read_csv(DATA_DIR / DATASET_NAME)
data.head()


# In[5]:


# Group by case_id and build sentence list + label list
def aggregate_case(group):
    sentences = group["sentence_text"].tolist()
    labels = [1 if rel in ["essential", "relevant"] else 0 for rel in group["relevance"]]
    return pd.Series({
        "question": group["patient_question"].iloc[0],
        "sentences": sentences,
        "labels": labels
    })

# Select only needed columns before grouping to silence the warning
data = (
    data[["case_id", "patient_question", "sentence_text", "relevance"]]
    .groupby("case_id")
    .apply(aggregate_case)
    .reset_index()
)


# In[6]:


data.head()


# In[7]:


data.iloc[0].sentences


# # Masking

# In[8]:


WINDOW_SIZE = 2


# In[9]:


test_df = util.mask_on_sentence_level(data, window=WINDOW_SIZE)


# In[10]:


test_df.head()


# # Model

# In[11]:


model_dir = Path("../../models")
model_name = "BioMedBert-W2"

model = AutoModelForSequenceClassification.from_pretrained(model_dir / model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_dir / model_name)


# In[12]:


model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device);


# # Prepare Dataset

# In[13]:


BATCH_SIZE = 64
CONTEXT_LENGTH = 512


# In[14]:


dataset_test = Dataset.from_pandas(test_df)


# In[15]:


progress_bar = tqdm(total=(len(dataset_test)),
                    desc="Tokenizing", position=0, leave=True)


# In[16]:


def tokenize_batch(batch):
    encodings = tokenizer(
        batch["question"],
        batch["context"],
        padding="max_length",
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_tensors="pt"
    )
    return {
        "input_ids": encodings["input_ids"].tolist(),
        "attention_mask": encodings["attention_mask"].tolist(),
        "labels": batch["label"]
    }

def tokenize_with_progress(batch):
    out = tokenize_batch(batch)
    progress_bar.update(len(batch["question"]))
    return out


# In[17]:


tokenized_dataset_test = dataset_test.map(tokenize_with_progress, batched=True, batch_size=BATCH_SIZE)


# In[18]:


progress_bar.close()


# In[19]:


tokenized_dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# In[20]:


test_dataloader = DataLoader(tokenized_dataset_test, batch_size=BATCH_SIZE)


# In[21]:


# Check one batch
batch = next(iter(test_dataloader))
print({key: value.shape for key, value in batch.items()})


# In[22]:


print("----- Test Set -----")
print(tokenized_dataset_test)
print(tokenized_dataset_test.column_names)


# # Evaluation

# In[29]:


all_preds = []
all_labels = []

threshold = 0.3

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"][:, 1].to(device)  # Get binary label (1 = relevant)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).squeeze()  # Shape: [batch_size]

        preds = (probs > threshold).long()

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

print(classification_report(all_labels, all_preds, digits=4, output_dict=True))


# In[27]:


print(np.array(all_labels).shape)
print(np.array(all_preds).shape)


# In[26]:


classification_report(all_labels, all_preds, digits=4, output_dict=True)


# In[24]:


report = pd.DataFrame(classification_report(all_labels, all_preds, digits=4, output_dict=True)).transpose()


# In[ ]:


display(report)


# **Window-Size: 5**

# In[77]:


display(report)


# **Window-Size: 3**

# In[53]:


display(report)


# **Window-Size: 2**

# In[28]:


display(report)


# **Window-Size: 1**

# In[160]:


display(report)


# In[ ]:




