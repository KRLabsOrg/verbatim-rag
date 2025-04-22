# Cross-Encoder for Clinical Question Answering

This directory contains scripts for training and using a cross-encoder model to identify relevant sentences in clinical notes based on a question.

## What is a Cross-Encoder?

A cross-encoder is a transformer-based model that evaluates the relevance between pairs of text (e.g., a question and a sentence). Unlike embedding models that embed texts into vectors independently, cross-encoders process the paired texts together through a shared neural network, resulting in one relevance score. This allows the model to better understand the relationship between the question and potential answer sentences.

## Workflow

1. **Data Preparation**: Process clinical notes and questions from the synthetic dataset, using the existing sentence segmentation and relevance labels.
2. **Model Training**: Train a cross-encoder model to classify sentences as relevant (1) or not relevant (0) to a clinical question.
3. **Sentence Ranking**: Use the trained model to rank sentences from clinical notes by relevance to new questions.

## Available Scripts

### Data Preparation

```bash
python scripts/archehr/prepare_crossencoder_data.py \
    --synthetic_data data/archehr/synthetic/questions/generated_questions_RUN02.csv \
    --output_dir data/crossencoder \
    --val_size 0.1
```

This script processes the synthetic questions dataset which contains pre-segmented sentences and their relevance labels. The script formats this data into paired examples (question, sentence, label) and splits them into training and validation sets. The output is saved as TSV files (train.tsv, val.tsv) suitable for training a cross-encoder model.

### Model Training

```bash
python scripts/archehr/train_crossencoder.py \
    --train_data data/crossencoder/train.tsv \
    --val_data data/crossencoder/val.tsv \
    --model_name KRLabsOrg/chiliground-base-modernbert-v1 \
    --output_dir models/crossencoder \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --eval_steps 500 \
    --fp16
```

This script trains a cross-encoder model on the prepared data. It uses the Sentence Transformers library and evaluates the model on the validation set at regular intervals. The best model checkpoints are saved during training, and a final model is saved upon completion.

### Sentence Ranking

```bash
python scripts/archehr/rank_sentences.py \
    --model models/crossencoder/YYYY-MM-DD_HH-MM-SS \
    --input data/archehr/synthetic/questions/generated_questions_RUN02.csv \
    --output results/crossencoder/ranked_sentences.json \
    --max_sentences 10 \
    --threshold 0.5
```

This script uses the trained model to rank sentences from clinical notes by relevance to questions. It uses the pre-segmented sentences from the input data and scores each one using the cross-encoder model.

## Model Selection

The default model used is `KRLabsOrg/chiliground-base-modernbert-v1`, which is a modern BERT-based model that performs well on clinical and biomedical text. You can also try other models such as:

- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- `microsoft/deberta-v3-base`
- `BAAI/bge-reranker-base`

## Data Format

The script expects the input data to have the following structure:
- `patient_question` or `clinician_question`: The question text
- `note_text`: The clinical note text, either as a list of sentences or as a JSON string representing a list
- `labels`: Binary relevance labels (0 or 1) for each sentence, matching the order of sentences

## References

- [Training and Finetuning Reranker Models with Sentence Transformers v4](https://huggingface.co/blog/train-reranker)
- [Sentence Transformers Documentation](https://www.sbert.net/) 