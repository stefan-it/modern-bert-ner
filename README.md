# ModernBERT NER Experiments

My NER Experiments with ModernBERT on the official CoNLL-2003 NER dataset.

# Results

Current results so far - at the moment with only one run with a specific configuration (= batch size, learning rate, nunber of epochs, context size).

All experiments are performed with latest Flair version, Micro F1-Score on the development set:

| Configuration           |   Run 1 | Avg.        |
|-------------------------|---------|-------------|
| `bs16-e10-cs0-lr3e-05`  |   95.78 | 95.78 ± 0.0 |
| `bs32-e10-cs0-lr3e-05`  |   95.46 | 95.46 ± 0.0 |
| `bs32-e10-cs0-lr5e-05`  |   95.45 | 95.45 ± 0.0 |
| `bs16-e10-cs0-lr5e-05`  |   95.32 | 95.32 ± 0.0 |
| `bs32-e10-cs0-lr0.0001` |   95.18 | 95.18 ± 0.0 |
| `bs4-e10-lr5e-06`       |   95.09 | 95.09 ± 0.0 |
| `bs16-e10-cs0-lr0.0001` |   94.41 | 94.41 ± 0.0 |

Performance is very low - I opened an issue about this [here](https://github.com/AnswerDotAI/ModernBERT/issues/149).

Watch and star this repo for updates.
