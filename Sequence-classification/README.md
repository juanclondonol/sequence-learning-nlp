# Sequence Classification: Academic Paper Categorization

This module focuses on training a fine-tuned transformer model (`distilbert-base-uncased`) to automatically categorize academic papers into specific scientific sub-disciplines using the first section of their text content.

The production model is deployed and accessible on the Hugging Face Hub:  
**Model Hub:** [jclondonol/Sequence_Classification_Models](https://huggingface.co/jclondonol/Sequence_Classification_Models)

---

## Dataset Description: `ccdv/arxiv-classification`

The dataset used for this task is hosted on Hugging Face: [`ccdv/arxiv-classification`](https://huggingface.co/datasets/ccdv/arxiv-classification).

### Domain
Scientific Literature and Academic Research.

This dataset is highly useful for automating the literature review process. It allows a model to read dense academic papers and automatically categorize them into their respective scientific disciplines. For engineering and research, it helps rapidly filter relevant theoretical or applied papers (e.g., isolating Computer Science literature from pure Mathematics) without manual reading.

### Problem Type
Sequence Classification (Multiclass Text Classification).

### Input Data
Raw text extracted from academic papers hosted on arXiv. Due to the length of the documents, the model typically processes the first 512 tokens (usually covering the title, authors, and abstract).

### Output
A single categorical label representing the academic sub-discipline of the paper. There are 11 distinct classes:

*   **`math.AC`**: Commutative Algebra
*   **`cs.CV`**: Computer Vision and Pattern Recognition
*   **`cs.AI`**: Artificial Intelligence
*   **`cs.SY`**: Systems and Control (Control Theory, Safety Certificates, Invariant Sets)
*   **`math.GR`**: Group Theory
*   **`cs.CE`**: Computational Engineering, Finance, and Science
*   **`cs.PL`**: Programming Languages
*   **`cs.IT`**: Information Theory
*   **`cs.DS`**: Data Structures and Algorithms
*   **`cs.NE`**: Neural and Evolutionary Computing
*   **`math.ST`**: Statistics Theory

---

## Dataset Structure

The dataset contains predefined splits for training, validation, and testing, structured as follows:

```text
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 28388
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 2500
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 2500
    })
})

```


### Text Preprocessing & Cleaning Pipeline
Academic papers usually begin with noisy metadata (author names, emails, dates, and institutional affiliations). Since the transformer's maximum sequence length is restricted to 512 tokens, keeping this metadata wastes valuable sequence space.

To maximize the density of scientific keywords, a regular expression pipeline splits the raw document text at the first occurrence of the keyword abstract (case-insensitive) and slices it to preserve only the text that follows.


### Pipeline Slicing Example

```
--- ORIGINAL TEXT (Starts with metadata) ---
Constrained Submodular Maximization via a
Non-symmetric Technique

arXiv:1611.03253v1 [cs.DS] 10 Nov 2016

Niv Buchbinder*
Moran Feldman†

November 11, 2016

Abstract
The study of combinatorial optimization problems with a submodular objective has attracted
much attention in recent years. Such problems are important in both theory and practice because
their objective functions are very general. Obtaining further improvements for many submodular
maximization problems boils down to finding better ...

--- CLEANED TEXT (Starts after 'Abstract') ---
The study of combinatorial optimization problems with a submodular objective has attracted
much attention in recent years. Such problems are important in both theory and practice because
their objective functions are very general. Obtaining further improvements for many submodular
maximization problems boils down to finding better algorithms for optimizing a relaxation of
them known as the multilinear extension.
In this work we present an algorithm for optimizing the multilinear relaxation whose ...
```


### Hyperparameter Tuning (Grid Search)

To find the optimal configuration for the Sequence Classification task using `distilbert-base-uncased`, a systematic grid search was conducted. The search utilized a representative subset of the dataset (1,000 training samples and 200 validation samples) evaluated across 5 epochs to maintain computational efficiency.

A total of 15 unique combinations of **learning rate** and **weight decay** were tested.

#### Search Space & Results

| Experiment | Learning Rate | Weight Decay | Validation Accuracy |
| :---: | :---: | :---: | :---: |
| 1 | `1e-5` | 0.001 | 0.7400 |
| 2 | `1e-5` | 0.01 | 0.7300 |
| 3 | `1e-5` | 0.1 | 0.7300 |
| 4 | `2e-5` | 0.001 | 0.7650 |
| 5 | `2e-5` | 0.01 | 0.7650 |
| 6 | `2e-5` | 0.1 | 0.7650 |
| 7 | `3e-5` | 0.001 | 0.7500 |
| 8 | `3e-5` | 0.01 | 0.7550 |
| 9 | `3e-5` | 0.1 | 0.7500 |
| 10 | `4e-5` | 0.001 | 0.7600 |
| 11 | `4e-5` | 0.01 | 0.7600 |
| 12 | `4e-5` | 0.1 | 0.7500 |
| 13 | `5e-5` | 0.001 | 0.7650 |
| **14 (Best)** | **`5e-5`** | **0.01** | **0.7700** |
| 15 | `5e-5` | 0.1 | 0.7600 |

#### Optimal Hyperparameters Summary

> **Best Configuration Found:**
> * **Learning Rate:** `5e-5`
> * **Weight Decay:** `0.01`
> * **Top Validation Accuracy:** **77.00%**

### Final Training & Production Evaluation
The optimal parameters found during the grid search (learning_rate=5e-5, weight_decay=0.01) were applied to a full-scale training run over the entire ccdv/arxiv-classification dataset.


### Training History Logs
| Epoch | Training Loss | Validation Loss | Accuracy |
| :---: | :---: | :---: | :---: |
| 1 | 0.688573 | 0.716916 | 0.7780 |
| 2 | 0.550924 | 0.669122 | 0.7968 |
| 3 | 0.420403 | 0.737860 | 0.8020 |
| 4 | 0.327788 | 0.843451 | 0.8016 |
| 5 | 0.246833 | 0.958078 | 0.7956 |


```Note: Since load_best_model_at_end=True was activated, the configuration corresponding to the lowest evaluation loss (Epoch 2) was preserved as the final operational model artifact.```



```
TrainOutput(
    global_step = 17745,
    total_training_loss = 0.472996,
    metrics = {
        'train_runtime': '3698.8815 seconds (~1h 01m 38s)',
        'train_samples_per_second': 38.374,
        'train_steps_per_second': 4.797,
        'total_flos': 1.880544039585792e+16
    }
)
```


### Production Real-World Inference Test

```
Limited capture range, and the requirement to provide high quality initialization for optimization-based 2D/3D image registration methods, can significantly degrade the performance of 3D image reconstruction and motion compensation pipelines. Challenging clinical imaging scenarios, which contain significant subject motion such as fetal in-utero imaging... 
In this paper we present a learning based image registration method capable of predicting 3D rigid transformations of arbitrarily oriented 2D image slices... We utilize a Convolutional Neural Network (CNN) architecture...
```

### Inference Output Execution Result

```py
from transformers import pipeline

# Instantiate production classifier directly from Hub checkpoint
classifier = pipeline(
    "text-classification", 
    model="jclondonol/Sequence_Classification_Models",
    truncation=True,
    max_length=512
)

print(classifier(test_text))
# Output: [{'label': 'cs.CV', 'score': 0.9495413303375244}]
```

The fine-tuned network mapped the diagnostic imaging paper sequence to the cs.CV (Computer Vision and Pattern Recognition) category with an explicit prediction confidence rating of 94.95%, validating correct inference classification routing.
