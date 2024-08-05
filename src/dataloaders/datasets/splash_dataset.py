import torch
import os
from pyfaidx import Fasta
import random
import pandas as pd

class SPLASHDatset(torch.utils.data.Dataset):

    '''
    Generic SPLASH dataset class for training and evaluation
    Including sep token option between anchor-target pairs
    
    '''

    def __init__(
        self,
        split,
        fasta_file,
        label_file,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        task = "classification|next_token_pred",
        replace_N_token=False,  # replace N token with pad token
        replace_X_token=True,  # replace X token with pad token
    ):

        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.replace_X_token = replace_X_token
        # assert os.path.exists(fasta_file), f"fasta file {fasta_file} not found"
        self.fasta = Fasta(fasta_file)
        self.task = task
        # check length of fasta file equal to split file
        # read split file which contains 2 columns split and label
        if label_file:
            assert os.path.exists(str(label_file)), f"label file {label_file} not found"
            df_label = pd.read_csv(label_file)
            assert len(df_label) == len(list(self.fasta)), f"Length of fasta file {len(list(self.fasta))} not equal to split file {len(df_label)}"

        else:
            # create a dummy label file with 80 -10 -10 split
            df_label = pd.DataFrame(columns=['split', 'label'])
            df_label['split'] = ['train'] * int(0.8 * len(self.fasta)) + ['val'] * int(0.1 * len(self.fasta)) + ['test'] * int(0.1 * len(self.fasta))
            df_label['label'] = [0] * len(self.fasta)
            #shuffle the split
            df_label = df_label.sample(frac=1).reset_index(drop=True)


        # get the indexes of the split
        chosen_indexes = df_label[df_label['split'] == split].index.tolist()
        # get the labels of the split
        self.labels = df_label[df_label['split'] == split]['label'].values
        
        assert len(chosen_indexes) > 0, f"Split {split} not found in split file"

        self.samples = []
        # get only sequence that are in the split
        for i, seq in enumerate(self.fasta):
            if i in chosen_indexes:
                self.samples.append(seq)

    def __len__(self):
        return len(self.samples)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random sequence from samples
        seq = self.samples[idx]
        seq = seq[:self.max_length]  # truncate to max_length
        seq = str(seq).upper()  # convert to uppercase

        if self.tokenizer_name == 'char':

            seq = self.tokenizer(seq,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)
        if self.replace_X_token:
            # replace X token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['X'], self.tokenizer.pad_token_id)

        if self.task == 'next_token_pred':
            target = seq[1:].clone()  # offset by 1, includes eos
        elif self.task == 'classification':
            target = self.labels[idx]
        else:
            raise ValueError(f"Invalid task: {self.task}")

        data = seq[:-1].clone()  # remove eos
        return data, target
