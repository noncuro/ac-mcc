"""
# Before starting:
```shell
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
```

Note: Some code for fine-tuning BERT is copied from https://huggingface.co/transformers/custom_datasets.html?highlight=read_imdb_split#seq-imdb

"""

from pathlib import Path
from typing import *

import numpy as np
import torch
import wandb
from scipy.stats import spearmanr
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels


def rand_indices_with_values(labels, requirements: Dict[Any, int]):
    # Requirements = {0:2, 1:3} means "I want 2 *unique* 0s and 3 unique 1s"
    """
    >>> labels = np.array(([0]*5) + ([1]*5))
    >>> res = rand_indices_with_values(labels, {0:2, 1:2})
    >>> len(labels[res])
    4
    >>> len(np.unique(res)[0])
    4
    >>> res.mean()
    0.5
    """
    labels = np.array(labels)
    res = []
    for k, n in requirements.items():
        idxs = np.random.choice(np.where(labels == k)[0], n, replace=False)
        res.append(idxs)
    return np.hstack(res)


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    train_texts, train_labels = read_imdb_split('aclImdb/train')
    test_texts, test_labels = read_imdb_split('aclImdb/test')

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.5)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    train_idxs = rand_indices_with_values(train_labels, {0: 1000, 1: 250})
    val_idxs = rand_indices_with_values(val_labels, {0: 1000, 1: 250})
    test_idxs = rand_indices_with_values(test_labels, {0: 250, 1: 1000})

    train_subset = Subset(train_dataset, train_idxs)
    val_subset = Subset(val_dataset, val_idxs)
    test_subset = Subset(test_dataset, test_idxs)

    for i in range(20):
        train_idxs = rand_indices_with_values(train_labels, {0: 1000, 1: 250})
        val_idxs = rand_indices_with_values(val_labels, {0: 1000, 1: 250})
        test_idxs = rand_indices_with_values(test_labels, {0: 250, 1: 1000})

        train_subset = Subset(train_dataset, train_idxs)
        val_subset = Subset(val_dataset, val_idxs)
        test_subset = Subset(test_dataset, test_idxs)

        training_args = TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=1,  # total number of training epochs
            per_device_train_batch_size=4,  # batch size per device during training
            per_device_eval_batch_size=4,  # batch size for evaluation
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=10,
            seed=i
        )

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_subset,  # training dataset
            eval_dataset=val_subset  # evaluation dataset
        )

        trainer.train()

        val_preds = trainer.predict(val_subset)

        test_preds = trainer.predict(test_subset)

        test_pred_mean = test_preds.predictions.argmax(1).mean()
        val_mcc = spearmanr(test_preds.predictions.argmax(1), test_preds.label_ids).correlation
        tpr, fpr = metrics.confusion_matrix(val_preds.label_ids, val_preds.predictions.argmax(1), labels=[1, 0],
                                            normalize='true')[:, 0]

        p_train = np.array(train_labels)[train_idxs].mean()
        p_val = np.array(val_labels)[val_idxs].mean()
        p_test = np.array(test_labels)[test_idxs].mean()

        ac_pred = (test_pred_mean - (1 - val_mcc) * p_train) / val_mcc

        mcc_pred = (test_pred_mean - fpr) / (tpr - fpr)

        wandb.log(dict(
            test_pred_mean=test_pred_mean,
            val_mcc=val_mcc,
            tpr=tpr, fpr=fpr,
            p_train=p_train,
            p_val=p_val,
            p_test=p_test,
            ac_pred=ac_pred,
            mcc_pred=mcc_pred
        ))
        wandb.finish()
