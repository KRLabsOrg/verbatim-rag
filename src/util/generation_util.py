import sys                              
import re                               
import ast                              
import time                             
import random                           
import logging                          
import argparse                         
from pathlib import Path                
import pandas as pd                     
import nltk                             
from tqdm import tqdm                   

import openai                           
import requests                         

import util.preprocessing_util as util


def load_prompt(template_path: Path) -> str:
    """
    Load the prompt template from disk.
    Raises if the file does not exist.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {template_path}")
    return template_path.read_text()


def format_few_shot_examples(df: pd.DataFrame) -> str:
    """
    Format a DataFrame of few-shot examples into a single prompt block.
    Each example includes a numbered note excerpt and its QA pair.
    """
    blocks = []
    for _, row in df.iterrows():
        # Number each sentence in the example note
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(row.sentences))
        # Collect indices of sentences labeled relevant
        relevant = [i+1 for i, lbl in enumerate(row.labels) if lbl]
        blocks.append(
            f"Note Excerpt:\n{numbered}\n\n"
            f"Patient Question: {row.question}\n"
            f"Clinician Question: {row.clinician_question}\n"
            f"Relevant Sentences: {relevant}"
        )
    # Join examples with separator
    return "\n\n---\n\n".join(blocks)


def is_valid_generation(text: str) -> bool:
    """
    Quick check that a generated block contains the expected fields.
    """
    t = text.lower()
    return (
        "patient question:" in t and 
        "clinician question:" in t and 
        "relevant sentences" in t and
        "[" in t and "]" in t
    )