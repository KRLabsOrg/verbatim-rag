#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import re
import json
import unicodedata
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("../../src")
import util.preprocessing_util as util


# In[3]:


DATA_DIR = Path("../../data/dev")
input_dir = DATA_DIR / "raw"
output_dir = DATA_DIR / "processed"
data_file_name = "archehr-qa.xml"
key_file_name = "archehr-qa_key.json"
mapping_file_name = "archehr-qa_mapping.json"


# # Extract & Organize

# **Key-File:**
# - Provides sentence-level answer annotations.
# - Labels answers as "essential" or "not-relevant".

# Load from json file

# In[4]:


key_file_path = input_dir / key_file_name
with open(key_file_path, "r") as f:
    key_data = json.load(f)


# Convert to dataframe

# In[5]:


structured_key_data = []

for case in key_data:
    case_id = case["case_id"]
    for answer in case["answers"]:
        structured_key_data.append({
            "case_id": case_id,
            "sentence_id": answer["sentence_id"],
            "relevance": answer["relevance"]
        })

# Create Pandas DataFrame
key_df = pd.DataFrame(structured_key_data)


# In[6]:


key_df.head()


# **Mapping-File:**
# - Maps case IDs to specific documents from MIMIC-III.
# - Shows where the context for each question is located.

# Load from json file

# In[7]:


mapping_file_path = input_dir / mapping_file_name
with open(mapping_file_path, "r") as f:
    mapping_data = json.load(f)


# Convert to dataframe

# In[8]:


structured_mapping_data = []

for case in mapping_data:
    structured_mapping_data.append({
        "case_id": case["case_id"],
        "document_id": case["document_id"],
        "document_source": case["document_source"]
    })

# Create Pandas DataFrame
mapping_df = pd.DataFrame(structured_mapping_data)


# In[9]:


mapping_df.head()


# **Train-set:**
# 
# Contains cases with unique ids.
# Stores questions, associated clinical context

# Load from XML file

# In[10]:


def parse_xml_to_dataframe(path_to_xml):
    tree = ET.parse(path_to_xml)
    root = tree.getroot()

    structured_data = []

    for case in root.findall("case"):
        case_id = case.attrib["id"]

        # Extract patient details
        patient_narrative = case.find("patient_narrative").text if case.find("patient_narrative") is not None else "No patient narrative"
        patient_question = case.find("patient_question/phrase").text if case.find("patient_question/phrase") is not None else "No patient question"
        clinician_question = case.find("clinician_question").text if case.find("clinician_question") is not None else "No clinician question"

        # Extract clinical note excerpts
        note_excerpt = case.find("note_excerpt").text if case.find("note_excerpt") is not None else "No note excerpt"

        # Extract sentence-level details from note excerpts
        for sentence in case.findall("note_excerpt_sentences/sentence"):
            sentence_id = sentence.attrib["id"]
            paragraph_id = sentence.attrib["paragraph_id"]
            start_char_index = sentence.attrib["start_char_index"]
            length = sentence.attrib["length"]
            sentence_text = sentence.text if sentence.text is not None else "No sentence text"

            structured_data.append({
                "case_id": case_id,
                "patient_narrative": patient_narrative,
                "patient_question": patient_question,
                "clinician_question": clinician_question,
                "note_excerpt": note_excerpt,
                "sentence_id": sentence_id,
                "sentence_text": sentence_text,
                "paragraph_id": paragraph_id,
                "start_char_index": start_char_index,
                "length": length
            })

    return pd.DataFrame(structured_data)


# In[11]:


data_file_path = input_dir / data_file_name
data_df = parse_xml_to_dataframe(data_file_path)


# Cast attributes to int for merging

# In[12]:


data_df["case_id"] = data_df["case_id"].astype(int)
data_df["sentence_id"] = data_df["sentence_id"].astype(int)
key_df["case_id"] = key_df["case_id"].astype(int)
key_df["sentence_id"] = key_df["sentence_id"].astype(int)
mapping_df["case_id"] = mapping_df["case_id"].astype(int)


# Merge the dataframes into one

# In[13]:


# Merge XML data with answer relevance labels
temp_df = data_df.merge(key_df, on=["case_id", "sentence_id"], how="left")

# Merge with document mapping
all_df = temp_df.merge(mapping_df, on="case_id", how="left")


# In[14]:


all_df.head()


# In[15]:


all_df.iloc[0].note_excerpt


# # Data View

# Routine check for missing values

# In[16]:


all_df.isna().sum()


# List categories of relevance

# In[17]:


all_df["relevance"].value_counts()


# List distribution of different documents

# In[18]:


all_df["document_id"].value_counts()


# In[19]:


test = all_df["note_excerpt"].unique()


# In[20]:


print(test[0])


# # Data Cleaning

# In[21]:


text_columns = ["patient_narrative", "patient_question", "clinician_question", "note_excerpt", "sentence_text"]


# In[22]:


cleaned_data = util.clean_text_df(all_df, text_columns, list_columns=[])


# In[23]:


cleaned_data.iloc[0].note_excerpt


# Save the processed data (using csv instead of pickle for now since dataset is fairly small)

# In[24]:


cleaned_data["case_id"].nunique()


# In[25]:


cleaned_data["note_excerpt"].nunique()


# # Attribute Selection 

# In[26]:


relevant_columns = ["case_id", "patient_question", "clinician_question", "note_excerpt", "sentence_id", "sentence_text", "relevance", "start_char_index", "length"]


# In[27]:


cleaned_data = cleaned_data[relevant_columns]


# In[28]:


cleaned_data["relevance"].value_counts()


# In[29]:


sns.countplot(x="relevance", data=cleaned_data)
plt.title("Class Distribution in 'relevance'")
plt.xlabel("Relevance")
plt.ylabel("Count")
plt.show()


# # Transform to right format 

# In[30]:


# Group by case_id and build sentence list + label list
def aggregate_case(group):
    sentences = group["sentence_text"].tolist()
    labels = [1 if rel in ["essential", "relevant"] else 0 for rel in group["relevance"]]
    return pd.Series({
        "patient_question": group["patient_question"].iloc[0],
        "clinician_question": group["clinician_question"].iloc[0],
        "note_excerpt": group["note_excerpt"].iloc[0],
        "sentences": sentences,
        "sentence_text": group["sentence_text"].iloc[0],
        "labels": labels
    })


# In[31]:


data_agg = (
    cleaned_data[["case_id", "patient_question", "clinician_question", "note_excerpt", "sentence_text", "relevance"]]
    .groupby("case_id")
    .apply(aggregate_case)
    .reset_index()
)


# **Quick sanity check**

# In[32]:


data_agg.head()


# **Save only if needed**

# In[33]:


'''
data_agg.to_csv(output_dir / "medical_data.csv", index=False)
print("Cleaned dataset saved successfully!")
''';


# # In-Depth Look at the Questions

# In[34]:


questions = data_agg[["patient_question", "clinician_question"]].drop_duplicates()


# In[45]:


for _, row in data_agg.iterrows():
    print(f"PQ: {row['patient_question']}")
    print(f"CQ: {row['clinician_question']}")
    print("\nNote Excerpt:")
    for i, tupel in enumerate(zip(row["sentences"], row["labels"])):
        print(f"{i+1}({tupel[1]}): {tupel[0]}")
    print()
    print("--------------------------------------------------------")
    print()


# In[ ]:




