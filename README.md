# hive-ml-classifier
An experiment to have a team of claudes generate data and fine tune a bert

## Pipeline

This repo is the ML side only: dataset generation (`data/`), training (`training/`), and model deliveries (`delivery*/`, config and tokenizer; the weights ship separately). The classifier routes text into four labels (`other`, `swarm`, `time`, `weather`) with a fully fine-tuned [prajjwal1/bert-mini](https://huggingface.co/prajjwal1/bert-mini). It needs `torch`, `transformers`, `datasets`, `scikit-learn`, `pandas` and `safetensors`.

```bash
cd data
python generate.py   # expand templates.json into dataset.csv
python split.py      # 80/20 split into train.csv and val.csv
cd ../training
python train.py      # writes training/output/ (gitignored), including model.safetensors
python eval.py       # evaluates on data/val.csv and a set of edge cases
```

`generate.py` reads `templates.json` from the current directory, so run it from `data/`. Training uses a max length of 64 tokens. `train.py` renames the `LayerNorm` gamma/beta tensors to weight/bias, which is the naming Candle and current Transformers expect. The number of labels is read from the delivered `config.json`, so a model with a different label set needs no code change on the consumer side.
