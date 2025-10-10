import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import os

system_instruction= "" REDACTED """

class MultiObjRMDataset(Dataset):
    def __init__(self, tokenizer=None, arrow_dir=None, max_length=20000, split='train',
                 precomputed_col=None, base_hidden_dir=None):
        """
        tokenizer        : HuggingFace tokenizer
        arrow_dir        : Path to directory containing split's Arrow file (data-00000-of-00001.arrow)
        split            : 'train' or 'test'
        precomputed_col  : Name of the column in dataset that contains hidden state file paths
        base_hidden_dir  : Base directory where the hidden state files are stored
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.precomputed_col = precomputed_col
        self.base_hidden_dir = base_hidden_dir

        # Load dataset split from Arrow file
        arrow_path = os.path.join(arrow_dir, "data-00000-of-00001.arrow")
        self.dataset = HFDataset.from_file(arrow_path)
        print(f"Loaded {len(self.dataset)} examples from {arrow_path}")

        # Calculate max sequence length if not using precomputed embeddings
        if not self.precomputed_col:
            max_seq_len = 0
            for example in self.dataset:
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": example["ocr_output"]},
                    {"role": "user", "content": example["question"]}
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                tokenized = self.tokenizer(text, truncation=False, add_special_tokens=True)
                length = len(tokenized["input_ids"])
                max_seq_len = max(max_seq_len, length)
            print(f"Maximum sequence length in dataset: {max_seq_len}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        labels = torch.tensor([
            int(example["low_effort"]),
            int(example["evidence"]),
            int(example["factual"])
        ], dtype=torch.long)

        if self.precomputed_col:
            # File path comes from the dataset column
            rel_path = example[self.precomputed_col]
            rep_path = os.path.join(self.base_hidden_dir, self.split, rel_path)
            rep = torch.load(rep_path)
            rep = rep.squeeze(0)
            return {"rep": rep, "labels": labels}

        # Tokenize if no precomputed embeddings
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": example["ocr_output"]},
            {"role": "user", "content": example["question"]}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels
        }
