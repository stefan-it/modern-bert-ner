# ModernBERT NER Experiments

My NER Experiments with ModernBERT on the official CoNLL-2003 NER dataset.

# Results I

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

# Results II

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

# Results III

I fixed the Tokenizer issue on the model side by forking the original ModernBERT Large model and applying the fixes in the Tokenizer config.
This model is available [here](https://huggingface.co/stefan-it/ModernBERT-large-tokenizer-fix) under the `stefan-it/ModernBERT-large-tokenizer-fix` Model Hub identifier.

Additionally, new experiments are conducted by pooling the first and last subtoken (instead of only using the first subtoken embedding), as this improved performance.
