# Protein Mutation Classification

This repository provides a framework for classifying protein sequence mutations as `Stable` or `Unstable` using embeddings extracted from a pretrained ESM model and a custom LSTM-based classifier with attention.

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
   git clone https://github.com/Rohit-VP/EmbNetMut-Integrating-Embedding-Features-with-Bi-LSTM-for-Protein-Mutation-Stability-Classification/tree/main
   cd EmbNetMut-Integrating-Embedding-Features-with-Bi-LSTM-for-Protein-Mutation-Stability-Classification
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

# Set the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained ESM model and tokenizer
model_dir = 'nmodel'
model = EsmModel.from_pretrained(model_dir).to(device)
tokenizer = EsmTokenizer.from_pretrained(model_dir)

# Define the input sequences
original_seq = 'SWIKEKKLL'  # Example original sequence
mutated_seq = 'SWIKAKKLL'   # Example mutated sequence

# Tokenize the sequences
inputs = tokenizer(original_seq, mutated_seq, return_tensors='pt', max_length=25, padding=True, truncation=True)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
```

### 2. Loading the Classifier

The classifier is a custom LSTM model designed to work with the extracted embeddings. Follow these steps:

1. Ensure the embeddings are stored in the correct format (numpy arrays, with dimensions matching the model's input requirements).
2. Define the `ComplexLSTM` class and initialize it:

```python
class ComplexLSTM(nn.Module):
    ...  # Define the architecture as provided in the code

model = ComplexLSTM()
```

3. Load the pretrained classifier weights:

```python
model_path = "best_model_nine.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
```

### 3. Inference

To classify a mutation as `Stable` or `Unstable`, use the extracted embeddings as input to the classifier:

1. Prepare the embeddings tensor:

```python
embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
```

2. Perform inference:

```python
with torch.no_grad():
    output = model(embedding_tensor)
    prediction = torch.argmax(output, dim=1).item()

if prediction == 1:
    print("Prediction: Forward")
else:
    print("Prediction: Reverse")
```

### Inputs and Outputs

#### Inputs:
- **Original Sequence (`original_seq`)**: The unmutated protein sequence.
- **Mutated Sequence (`mutated_seq`)**: The protein sequence after mutation.

#### Outputs:
- A prediction indicating whether the mutation is classified as `Stable` (1) or `Unstable` (0).


## Troubleshooting

1. **Device not recognized**: Ensure PyTorch is installed with GPU support if running on CUDA.
2. **Tokenization errors**: Verify the sequences and tokenizer compatibility.

