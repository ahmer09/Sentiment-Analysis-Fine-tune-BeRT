import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from tqdm import tqdm


BATCH_SIZE = 8
MAX_SEQUENCE_LENGTH = 128

class Dataset:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def convert_example(self, example):
        return torch.tensor(self.tokenizer.encode(example, add_special_tokens=True))

    def convert_to_dataloader_eval(self, df, text_field='SentimentText', label_field='labels'):
        """Loads a data field in Torch Dataset"""
        num_samples = len(df)
        all_segment_ids = torch.zeros((num_samples, MAX_SEQUENCE_LENGTH), dtype=torch.long)
        all_input_ids = torch.zeros((num_samples, MAX_SEQUENCE_LENGTH), dtype=torch.long)
        all_imput_mask = torch.zeros((num_samples, MAX_SEQUENCE_LENGTH), dtype=torch.long)
        all_label_ids = torch.empty((num_samples, 1), dtype=torch.long)
        for index, row in tqdm(df.iterrows(), total=num_samples, desc="Converting: "):
            tokens = self.tokenizer.tokenize(row[text_field])
            if len(tokens) > MAX_SEQUENCE_LENGTH - 2:
                tokens = tokens[:(MAX_SEQUENCE_LENGTH - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            all_input_ids[index, :len(input_ids)] = torch.tensor(input_ids)
            all_imput_mask[index, :len(input_ids)] = torch.Tensor([1] * len(input_ids))
            all_label_ids[index] = row[label_field]
        data = TensorDataset(all_input_ids, all_imput_mask, all_segment_ids, all_label_ids)
        dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=BATCH_SIZE)
        return dataloader

    def convert_to_dataloader_train(self, df, text_field='text', label_field='labels'):
        """Loads a data field in Torch Dataset"""
        num_samples = len(df)
        all_segment_ids = torch.zeros((num_samples, MAX_SEQUENCE_LENGTH), dtype=torch.long)
        all_input_ids = torch.zeros((num_samples, MAX_SEQUENCE_LENGTH), dtype=torch.long)
        all_imput_mask = torch.zeros((num_samples, MAX_SEQUENCE_LENGTH), dtype=torch.long)
        all_label_ids = torch.empty((num_samples, 1), dtype=torch.long)
        for index, row in tqdm(df.iterrows(), total=num_samples, desc="Converting: "):
            tokens = self.tokenizer.tokenize(row[text_field])
            if len(tokens) > MAX_SEQUENCE_LENGTH - 2:
                tokens = tokens[:(MAX_SEQUENCE_LENGTH - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            all_input_ids[index, :len(input_ids)] = torch.tensor(input_ids)
            all_imput_mask[index, :len(input_ids)] = torch.Tensor([1] * len(input_ids))
            all_label_ids[index] = row[label_field]
        data = TensorDataset(all_input_ids, all_imput_mask, all_segment_ids, all_label_ids)
        dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=BATCH_SIZE)
        return dataloader