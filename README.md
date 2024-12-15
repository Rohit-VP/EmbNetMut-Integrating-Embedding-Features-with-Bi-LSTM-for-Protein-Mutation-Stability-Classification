
# Protein Mutation Classification

This repository provides a framework for classifying protein sequence mutations as `Forward` or `Reverse` using embeddings extracted from a pretrained ESM model and a custom LSTM-based classifier with attention.

## Features

- **Device Compatibility**: Runs on both CPU and GPU.
- **Embedding Extraction**: Uses Hugging Face's ESM model to generate sequence embeddings.
- **Custom Classifier**: Employs a bidirectional LSTM with attention for sequence classification.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- PyTorch
- Transformers (Hugging Face library)
- NumPy
- scikit-learn

## Setup

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install dependencies:

   ```bash
   pip install torch transformers numpy scikit-learn
   ```

3. Download or prepare the pretrained ESM model and place it in the `nmodel/` directory.

## Usage

### 1. Embedding Extraction

The following steps extract embeddings for a protein sequence mutation:

1. Define the original and mutated sequences.
2. Load the ESM model and tokenizer.
3. Generate embeddings using the following script:

```python
import torch
from transformers import EsmModel, EsmTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = 'nmodel'
model = EsmModel.from_pretrained(model_dir).to(device)
tokenizer = EsmTokenizer.from_pretrained(model_dir)

original_seq = 'SWIKEKKLL'
mutated_seq = 'SWIKAKKLL'
inputs = tokenizer(original_seq, mutated_seq, return_tensors='pt', max_length=25, padding=True, truncation=True)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
```

### 2. Loading the Classifier

Train the custom LSTM-based classifier using the provided dataset:

1. Ensure the data is prepared in the format `(embedding, label)`.
2. Use the `ComplexLSTM` and `Attention` classes for model definition.

```python
class ComplexLSTM(nn.Module):
    ...  # Define the architecture as in the code

model = ComplexLSTM()
```

### 3. Inference

To classify a mutation as `Stable` or `Unstable`:

1. Load the pretrained classifier:

```python
model_path = "best_model_nine.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
```


## Troubleshooting

1. **Device not recognized**: Ensure PyTorch is installed with GPU support if running on CUDA.
2. **Tokenization errors**: Verify the sequences and tokenizer compatibility.
