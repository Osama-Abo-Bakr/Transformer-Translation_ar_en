# Transformer Translation (Arabic to English)

This repository contains a project focused on building a machine translation model that translates Arabic text to English using the Helsinki-NLP `opus-mt-ar-en` model. The implementation utilizes Hugging Face's Transformers library and the Datasets library for data preparation, training, and evaluation.

## Repository Details
- **Repository Name**: Transformer-Translation_ar_en
- **Repository Link**: [GitHub Link](https://github.com/Osama-Abo-Bakr/Transformer-Translation_ar_en)

## Features
- Implements a pre-trained transformer model for machine translation.
- Uses the Hugging Face Transformers library for model fine-tuning.
- Includes BLEU score evaluation for measuring translation quality.

## Installation
To get started, install the required libraries:

```bash
!pip install transformers datasets evaluate sacrebleu
```

## Project Workflow

### 1. Data Preparation
The dataset used for training and testing is the `Helsinki-NLP/un_ga` dataset.

```python
from datasets import load_dataset

data = load_dataset("Helsinki-NLP/un_ga", "ar_to_en")
data = data['train'].train_test_split(test_size=0.2)
```

### 2. Preprocessing
The preprocessing step involves:
- Tokenizing Arabic text with a prefix "translate Arabic to English:".

```python
from transformers import AutoTokenizer

prefix = "translate Arabic to English: "
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

def preprocess_function(examples):
    inputs = [prefix + example["ar"] for example in examples["translation"]]
    targets = [example["en"] for example in examples["translation"]]
    return tokenizer(inputs, text_target=targets, max_length=128, truncation=True)

tokenized_data = data.map(preprocess_function, batched=True)
```

### 3. Model Setup
The pre-trained Helsinki-NLP model is used for fine-tuning:

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
```

### 4. Training
Fine-tuning the model with Hugging Face's `Seq2SeqTrainer`:

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./Helsinki-mt-ar-en",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    num_train_epochs=1,
    fp16=True,
    warmup_steps=2000,
    logging_steps=2000
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
```

### 5. Evaluation
Evaluation of the model using BLEU score:

```python
import evaluate

metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}
```

### 6. Translation Testing
To make predictions on new Arabic text:

```python
def predict(text, model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    inputs = tokenizer(text, return_tensors="pt").input_ids
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

text = "\u0645\u0631\u062d\u0628\u0627 \u0643\u064a\u0641 \u062d\u0627\u0644\u0643؟"
print(predict(text, './Helsinki-mt-ar-en'))
```

## Results
The fine-tuned model was evaluated using the BLEU metric to measure translation quality. Users can further refine the model by adjusting hyperparameters or training epochs.

## References
- [Helsinki-NLP Models](https://huggingface.co/Helsinki-NLP)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Datasets Library](https://huggingface.co/docs/datasets)

---
For more details, check the [repository link](https://github.com/Osama-Abo-Bakr/Transformer-Translation_ar_en).

