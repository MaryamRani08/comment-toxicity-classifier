# Comment Toxicity Classifier

A multi-label text classification project built with TensorFlow/Keras and an interactive Gradio interface.

The application accepts an English comment and predicts whether it belongs to six toxicity categories. A comment can receive multiple labels.

## Categories

- Toxic
- Severe toxic
- Obscene
- Threat
- Insult
- Identity hate

## Features

- Text preprocessing using Keras TextVectorization
- Bidirectional LSTM classification model
- Predictions for individual comments and batches
- Precision and recall evaluation
- Model saving and loading
- Interactive Gradio demonstration

The interface displays True or False for each category using a prediction threshold of 0.5.

## Dataset

The project uses the Jigsaw Toxic Comment Classification Challenge dataset. The loaded training CSV contains 159,571 comments with six binary labels.

Download the data from:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

The dataset is not included in this repository.

## Architecture

| Component | Configuration |
|---|---|
| TextVectorization | Maximum 200,000 tokens; sequence length 1,800 |
| Embedding | 32 dimensions |
| Bidirectional LSTM | 32 units per direction |
| Dense layers | 128 → 256 → 128 units, ReLU |
| Output | 6 units, sigmoid |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Batch size | 16 |

The notebook is configured for an initial training run of one epoch.

### Architecture Rationale

The embedding layer learns token representations. The bidirectional LSTM processes text in both directions to capture sequential context.

Six sigmoid outputs allow multiple labels per comment, with binary cross-entropy used to train these binary predictions. Comparative experiments are planned to assess whether this architecture is appropriate relative to simpler and pretrained alternatives.

## Technologies

Python, TensorFlow/Keras, Pandas, NumPy, Matplotlib, JupyterLab, and Gradio.

## Getting Started

1. Clone the repository and activate a Python environment.
2. Install the notebook dependencies:

```bash
python -m pip install tensorflow pandas numpy matplotlib scikit-learn gradio jinja2
```

3. Download and extract the dataset.
4. Open `Toxicity.ipynb` in JupyterLab.
5. Update the CSV path to match your local dataset location.
6. Run the preprocessing and model sections.
7. Run the Gradio cells to launch the demonstration.

The current notebook reads the dataset from:

```python
df = pd.read_csv(
    os.path.join(
        "jigsaw-toxic-comment-classification-challenge",
        "train.csv",
        "train.csv"
    )
)
```

Adjust this path if your extracted folder structure differs.

The vocabulary inspection cell, `vectorizer.get_vocabulary()`, is optional and can be skipped to avoid unnecessary memory usage.

## Saving and Loading

The notebook saves the classifier using:

```python
model.save("toxicity_1.keras")
```

Load an existing saved model with:

```python
model = tf.keras.models.load_model("toxicity_1.keras")
```

When loading a trained model, skip the cells that recreate or retrain it.

The text vectorizer is separate from the saved classifier. Inference requires the same vocabulary, token ordering, and preprocessing configuration used during training. Packaging the vectorizer with the model is a planned improvement.

## Interactive Demo

The Gradio interface calls `score_comment()` and displays the six predicted labels.

To run locally:

```python
interface.launch()
```

The notebook currently uses `share=True` to create a temporary public demo link. This is not permanent deployment.

## Evaluation Status

The notebook includes an initial evaluation loop, but its results are preliminary.

Two issues have been identified:

- The current shuffle-before-split pipeline does not guarantee separate training, validation, and test sets.
- Categorical accuracy on flattened multi-label predictions is not an appropriate accuracy measure.

The vectorizer is also currently adapted on all comments rather than only the training split.

These issues will be corrected before publishing benchmark results. The current demonstration should be treated as a learning prototype.

## Planned Improvements

- Create fixed, non-overlapping training, validation, and test sets.
- Adapt text preprocessing only on training data.
- Report per-label precision, recall, and F1-score.
- Experiment with 5–7 epochs using early stopping and best-model checkpoints.
- Investigate sequence length and memory usage.
- Analyze false positives, false negatives, and class imbalance.
- Compare with a TF-IDF and logistic regression baseline.
- Explore a pretrained transformer such as DistilBERT.
- Package the vectorizer and classifier for reproducible inference.

## Limitations

The model may misclassify sarcasm, quotations, ambiguous comments, and unfamiliar language. It may also reflect biases in its training data.

It has not been validated for production moderation or decisions without human review.
