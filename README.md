# ModernBERT/Ettin NER Experiments

My NER Experiments with ModernBERT and Ettin on the official CoNLL-2003 NER dataset.

# ModernBERT

## Results I

Current results so far - at the moment with only one run with a specific configuration (= batch size, learning rate, nunber of epochs, context size).

All experiments are performed with latest Flair version, Micro F1-Score on the development set:

| Configuration           |   Run 1 | Avg.        |
|-------------------------|---------|-------------|
| `bs16-e10-cs0-lr3e-05`  |   95.78 | 95.78 ± 0.0 |
| `bs32-e10-cs0-lr3e-05`  |   95.46 | 95.46 ± 0.0 |
| `bs32-e10-cs0-lr5e-05`  |   95.45 | 95.45 ± 0.0 |
| `bs16-e10-cs0-lr5e-05`  |   95.32 | 95.32 ± 0.0 |
| `bs32-e10-cs0-lr0.0001` |   95.18 | 95.18 ± 0.0 |
| `bs4-e10-cs0-lr5e-06`   |   95.09 | 95.09 ± 0.0 |
| `bs16-e10-cs0-lr0.0001` |   94.41 | 94.41 ± 0.0 |

Performance is currently very low - I opened an issue about this [here](https://github.com/AnswerDotAI/ModernBERT/issues/149).

## Results II

After some debugging, it seems that the original tokenizer prepends a whitespace when performing the `tokenizer()` call in combination
with the `is_split_into_words` option (which is needed for token classification tasks).

One quick workaround was tested: using the `RobertaTokenizerFast` with `add_prefix_space=True`. For this Flair needs to be slighly modified:

```diff
diff --git a/flair/embeddings/transformer.py b/flair/embeddings/transformer.py
index fdb16eea2..09877a766 100644
--- a/flair/embeddings/transformer.py
+++ b/flair/embeddings/transformer.py
@@ -1080,9 +1080,17 @@ class TransformerEmbeddings(TransformerBaseEmbeddings):
 
         if tokenizer_data is None:
             # load tokenizer and transformer model
-            self.tokenizer = AutoTokenizer.from_pretrained(
-                model, add_prefix_space=True, **transformers_tokenizer_kwargs, **kwargs
-            )
+            if "modernbert" in model.lower():
+                from transformers import RobertaTokenizerFast
+                print("ModernBERT detected, using RoBERTa tokenizer!")
+                self.tokenizer = RobertaTokenizerFast.from_pretrained(
+                    model, add_prefix_space=True, **transformers_tokenizer_kwargs, **kwargs
+                )
+                self.tokenizer.model_max_length = 512
+            else:
+                self.tokenizer = AutoTokenizer.from_pretrained(
+                    model, add_prefix_space=True, **transformers_tokenizer_kwargs, **kwargs
+                )
             try:
                 self.feature_extractor = AutoFeatureExtractor.from_pretrained(model, apply_ocr=False, **kwargs)
             except OSError:
```

After this fix we get better performance:

| Configuration           |   Run 1 | Avg.        |
|-------------------------|---------|-------------|
| `bs16-e10-cs0-lr2e-05`  |   96.44 | 96.44 ± 0.0 |
| `bs16-e10-cs0-lr3e-05`  |   96.24 | 96.24 ± 0.0 |
| `bs16-e10-cs0-lr5e-05`  |   96.14 | 96.14 ± 0.0 |
| `bs16-e10-cs0-lr1e-05`  |   96.1  | 96.1 ± 0.0  |
| `bs16-e10-cs0-lr4e-05`  |   96.1  | 96.1 ± 0.0  |
| `bs16-e10-cs0-lr8e-05`  |   95.59 | 95.59 ± 0.0 |
| `bs16-e10-cs0-lr0.0001` |   95.37 | 95.37 ± 0.0 |
| `bs16-e10-cs0-lr0.0005` |   91.65 | 91.65 ± 0.0 |
| `bs16-e10-cs0-lr0.001`  |   89.2  | 89.2 ± 0.0  |
| `bs16-e10-cs0-lr0.005`  |   84.92 | 84.92 ± 0.0 |

More runs:

| Configuration          |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------|---------|---------|---------|---------|---------|--------------|
| `bs16-e10-cs0-lr2e-05` |   96.32 |   96.52 |   96.44 |   96.33 |   96.46 | 96.41 ± 0.08 |
| `bs16-e10-cs0-lr3e-05` |   96.29 |   96.24 |   96.39 |    96.2 |   96.14 | 96.25 ± 0.08 |
| `bs16-e10-cs0-lr4e-05` |   96.15 |   96.15 |    96.2 |   95.97 |    96.1 | 96.11 ± 0.08 |
| `bs16-e10-cs0-lr1e-05` |   95.93 |   96.13 |   96.06 |   96.05 |    96.1 | 96.05 ± 0.07 |
| `bs16-e10-cs0-lr5e-05` |   96.14 |   95.83 |   96.17 |   96.08 |   96.05 | 96.05 ± 0.12 |

Please watch and star this repo for updates.

## Results III

I fixed the Tokenizer issue on the model side by forking the original ModernBERT Large model and applying the fixes in the Tokenizer config.
This model is available [here](https://huggingface.co/stefan-it/ModernBERT-large-tokenizer-fix) under the `stefan-it/ModernBERT-large-tokenizer-fix` Model Hub identifier.
Using this new model has the huge advantage, that fixes in the Flair library are no longer needed!

Additionally, new experiments are conducted by pooling the first and last subtoken (instead of only using the first subtoken embedding), as this improved performance.

Here are new runs, with latest Flair and Transformers version:

| Configuration          | Pooling      |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------|--------------|---------|---------|---------|---------|---------|--------------|
| `bs16-e10-cs0-lr2e-05` | `first`      |   96.13 |   96.44 |   96.20 |   95.93 |   96.65 | 96.27 ± 0.25 |
| `bs16-e10-cs0-lr2e-05` | `first_last` |   96.36 |   96.58 |   96.14 |   96.19 |   96.35 | 96.32 ± 0.15 |

Results on the test set (also Micro F1-Score):

| Configuration          |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------|---------|---------|---------|---------|---------|--------------|
| `bs16-e10-cs0-lr2e-05` |   92.22 |   91.92 |   92.31 |   92.35 |   92.34 | 92.23 ± 0.16 |

# Ettin

The same hyper-parameter search is used for the recently released Ettin series. Ettin uses the same tokenizer as ModernBERT, so the exact same tokenizer fix is applied - the forked model can be found [here](https://huggingface.co/stefan-it/ettin-encoder-400m-tokenizer-fix)

## Results I

The `first_last` pooling strategy is also used - here are the results:

| Configuration          |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------|---------|---------|---------|---------|---------|--------------|
| `bs16-e10-cs0-lr4e-05` |   96    |   96.17 |   96.31 |   96.19 |   96.2  | 96.17 ± 0.1  |
| `bs16-e10-cs0-lr3e-05` |   96.25 |   96.23 |   96.12 |   96.3  |   95.81 | 96.14 ± 0.18 |
| `bs16-e10-cs0-lr2e-05` |   96.09 |   96.24 |   95.88 |   96.1  |   96.12 | 96.09 ± 0.12 |
| `bs16-e10-cs0-lr5e-05` |   95.98 |   95.93 |   96.11 |   96.1  |   96    | 96.02 ± 0.07 |
| `bs16-e10-cs0-lr1e-05` |   95.77 |   95.8  |   96.14 |   96.01 |   95.84 | 95.91 ± 0.14 |

Results on the test set (Micro F1-Score):

| Configuration          |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------|---------|---------|---------|---------|---------|--------------|
| `bs16-e10-cs0-lr4e-05` |   92.25 |   91.72 |   91.98 |   92.08 |    92.3 | 92.07 ± 0.21 |
