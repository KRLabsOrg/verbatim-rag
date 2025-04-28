## ChiliGround ArcheHR

This repository contains the code for the ChiliGround ArcheHR project.

## Process

```bash
python preprocess.py
```

It will process the data and save the processed data to `data/archehr/dev/archehr-qa_processed.json`.

### Predict

```bash
python archehr/predict.py --data_dir data/archehr/dev/archehr-qa_processed.json --model LLMModel
```


### Generate

TODO
