## Token Classification: Scientific Named Entity Recognition (SciNER)

This module focuses on training a fine-tuned transformer model (distilbert-base-uncased) to automatically extract domain-specific scientific entities, mathematical expressions, and engineering assets directly from the text body of peer-reviewed astrophysics publications.

The production model is deployed and accessible on the Hugging Face Hub:

Model Hub: [jclondonol/Token_Classification_Models](https://huggingface.co/jclondonol/nasa-astrophysics-ner-final)

## Dataset Description: adsabs/WIESP2022-NER

The dataset used for this task is provided by the NASA Astrophysical Data System (ADS) and hosted on Hugging Face: adsabs/WIESP2022-NER.

[Dataset](https://huggingface.co/datasets/adsabs/WIESP2022-NER)

### Domain
Astrophysics, Aerospace Systems, and Advanced Scientific Instrumentation Literature.

This dataset is instrumental for automated knowledge graph compilation and data extraction within highly technical domains. Instead of classifying a document as a whole, this module performs surgical, token-level extraction to isolate experimental hardware, software dependencies, physical formulas, and celestial targets. For engineering pipelines, it transforms unstructured scientific prose into operational databases without manual curation.

### Problem Type
Token Classification / Named Entity Recognition (Sequence Labeling via IOB2 Schema).

### Input Data
Pre-tokenized textual sequences extracted directly from full-text astrophysics articles. The inputs are structured as raw arrays of text strings (tokens) along with their corresponding positional text labels.

### Output
A synchronized sequence of classification labels mapping every individual word to its specific scientific category. The dataset tracks 63 distinct target labels using the strict IOB2 (B- Beginning, I- Inside, O Outside) structural syntax. Key target classes include:
- CelestialObject: Stars, black holes, galaxies, and planetary entities (e.g., Sgr A*, Exoplanet).
- Telescope: Ground-based or orbital astronomical arrays (e.g., CFHT, Hubble).
- Instrument: Spectrographs, cameras, and measurement sensors (e.g., ESPaDOnS).
- Formula: Mathematical equations and constant parameters (e.g., $GM/r^2$).
- Software: Coding languages, computational frameworks, and software packages (e.g., Python, NumPy).
- Grant: Research funding bodies, institutional codes, and sponsorship indices (e.g., 19MNRAS).

### Dataset Structure

The dataset contains predefined splits for training and validation, provided in a JSON Lines layout and structured as follows:
```
DatasetDict({
    train: Dataset({
        features: ['unique_id', 'tokens', 'ner_tags', 'ner_ids', 'label_studio_id', 'section', 'bibcode'],
        num_rows: 1753
    })
    validation: Dataset({
        features: ['unique_id', 'tokens', 'ner_tags', 'ner_ids', 'label_studio_id', 'section', 'bibcode'],
        num_rows: 1366
    })
})
```


### Hyperparameter Tuning (Grid Search)

To find the optimal configuration for the Token Classification task using distilbert-base-uncased, a systematic grid search was conducted. The search utilized a representative subset of the NASA/ADS dataset (400 training samples and 100 validation samples) evaluated across 3 epochs to effectively gauge learning trajectories while maintaining computational efficiency.

| Experiment | Learning Rate | Weight Decay | Eval Accuracy | Eval Precision | Eval Recall | Validation F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `1e-5` | 0.001 | 0.8693 | 0.2598 | 0.1184 | 0.1626 |
| 2 | `1e-5` | 0.01 | 0.8690 | 0.2779 | 0.1022 | 0.1494 |
| 3 | `1e-5` | 0.1 | 0.8690 | 0.2779 | 0.1022 | 0.1494 |
| 4 | `2e-5` | 0.001 | 0.8966 | 0.3252 | 0.2679 | 0.2938 |
| 5 | `2e-5` | 0.01 | 0.8966 | 0.3252 | 0.2679 | 0.2938 |
| 6 | `2e-5` | 0.1 | 0.8966 | 0.3252 | 0.2679 | 0.2938 |
| 7 | `3e-5` | 0.001 | 0.9176 | 0.4471 | 0.4070 | 0.4261 |
| 8 | `3e-5` | 0.01 | 0.9176 | 0.4471 | 0.4070 | 0.4261 |
| 9 | `3e-5` | 0.1 | 0.9176 | 0.4471 | 0.4070 | 0.4261 |
| 10 | `4e-5` | 0.001 | 0.9266 | 0.4982 | 0.4717 | 0.4846 |
| 11 | `4e-5` | 0.01 | 0.9266 | 0.4982 | 0.4717 | 0.4846 |
| 12 | `4e-5` | 0.1 | 0.9266 | 0.4982 | 0.4717 | 0.4846 |
| **13 (Best)** | **`5e-5`** | **0.001** | **0.9292** | **0.5101** | **0.4942** | **0.5021** |
| 14 | `5e-5` | 0.01 | 0.9292 | 0.5101 | 0.4942 | 0.5021 |
| 15 | `5e-5` | 0.1 | 0.9292 | 0.5098 | 0.4942 | 0.5019 |


### Optimal Hyperparameters Summary


> **Best Configuration Found:**
> * **Learning Rate:** `5e-5`
> * **Weight Decay:** `0.001`
> * **Top Validation F1-Score:** **50.21%**
> * **Top Validation Accuracy:** **92.92%**


### Final Training & Production Evaluation

The optimal parameters discovered during the hyperparameter grid search (learning_rate=5e-5, weight_decay=0.001) were applied to a full-scale training run over 100% of the adsabs/WIESP2022-NER dataset.

As predicted, training on the entire corpus resolved the data sparsity issues caused by the 63-class distribution layout. The increased structural exposure allowed the model to successfully generalize the semantic boundaries of rare technical entities, pushing the strict validation F1-Score from 50.21% up to a top performance of 76.46%.

### Training History Logs

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.364959 | 0.281082 | 0.596830 | 0.577838 | 0.587180 | 0.935052 |
| 2 | 0.210529 | 0.191051 | 0.666078 | 0.702459 | 0.683785 | 0.952570 |
| 3 | 0.140569 | 0.170836 | 0.680029 | 0.752402 | 0.714387 | 0.955210 |
| 4 | 0.113576 | **0.156613** | 0.726604 | 0.768733 | 0.747075 | 0.960474 |
| 5 | 0.081179 | 0.160400 | 0.733403 | 0.770085 | 0.751297 | 0.960801 |
| 6 | 0.066855 | 0.161836 | 0.743678 | 0.771473 | 0.757321 | 0.961468 |
| 7 | 0.061965 | 0.161179 | 0.731116 | 0.784005 | 0.756638 | 0.961288 |
| 8 | 0.050530 | 0.163752 | 0.743118 | 0.786014 | 0.763964 | 0.961854 |
| **9 (Best)** | **0.044264** | **0.167005** | **0.743774** | **0.786709** | **0.764639** | **0.962268** |
| 10 | 0.043908 | 0.167808 | 0.744507 | 0.783676 | 0.763590 | 0.962230 |


```
Note: Because 'load_best_model_at_end=True' was monitored alongside the 'f1' evaluation metric, the system preserved the exact artifact checkpoint from Epoch 9. This ensures maximum extraction precision and sequence boundary recognition on out-of-distribution academic literature.
```