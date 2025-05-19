import copy

import pandas as pd

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def cot_prompt_pre(src, system_prompt="You are a helpful assistant that provides implementation ideas for code."):
    """
    Custom pre-instruction template, now in chat format.
    :param src: The source code input from the user.
    :param system_prompt: An optional system prompt.
    :return: A list of dictionaries representing the chat messages.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": src}
    ]
    return messages

class GPTDataset(Dataset):
    """
    A custom dataset class for GPT model fine-tuning.
    It tokenizes source and target text sequences from a CSV file.
    """
    def __init__(self, filename, tokenizer, source_len, cutoff_len, system_prompt_content=None, nrows=None):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.cutoff_len = cutoff_len
        self.system_prompt_content = system_prompt_content

        # Determine the appropriate file reading function based on the file extension
        if filename.endswith(".csv"):
            self.datas = pd.read_csv(filename, nrows=nrows)
        elif filename.endswith(".jsonl"):
            self.datas = pd.read_json(filename, lines=True, nrows=nrows)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .jsonl files.")

        self.inputs = []
        self.token_labels = []

        length = len(self.datas)

        for idx in tqdm(range(length)):
            src = self.datas["src"][idx]
            tgt = self.datas["tgt"][idx]

            input_ids, input_labels = self.tokenize_prompt(src, tgt)
            self.inputs.append(input_ids)
            self.token_labels.append(input_labels)

    def tokenize_prompt(self, src, tgt):
        messages = []
        if self.system_prompt_content:
            messages.append({"role": "system", "content": self.system_prompt_content})
        messages.append({"role": "user", "content": src})
        messages.append({"role": "assistant", "content": tgt})

        tokenized_full_ids = self.tokenizer.apply_chat_template(
            messages,
            max_length=self.cutoff_len,
            truncation=True,
            padding=False, 
            add_generation_prompt=False,
            return_tensors=None 
        )
        input_ids = tokenized_full_ids
        labels = list(input_ids)

        prompt_messages_for_masking = []
        if self.system_prompt_content:
            prompt_messages_for_masking.append({"role": "system", "content": self.system_prompt_content})
        prompt_messages_for_masking.append({"role": "user", "content": src})
        
        tokenized_prompt_part_ids = self.tokenizer.apply_chat_template(
            prompt_messages_for_masking,
            max_length=self.cutoff_len, 
            truncation=True,
            padding=False,
            add_generation_prompt=True,
            return_tensors=None
        )
        prompt_length = len(tokenized_prompt_part_ids)

        for i in range(prompt_length):
            if i < len(labels):
                 labels[i] = -100
            else:
                 break 
        
        if all(label == -100 for label in labels):
            pass

        assert len(input_ids) == len(labels), "Input IDs and Labels must have the same length."
        return input_ids, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])