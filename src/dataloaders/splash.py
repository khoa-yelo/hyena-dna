import torch
import os
from pyfaidx import Fasta
import random
from genomics import HG38
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
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
        assert os.path.exists(fasta_file), f"fasta file {fasta_file} not found"
        self.fasta = Fasta(fasta_file)
        self.task = task
        # check length of fasta file equal to split file
        # read split file which contains 2 columns split and label
        if label_file:
            assert os.path.exists(str(label_file)), f"label file {label_file} not found"
            df_label = pd.read_csv(label_file)
            assert len(df_label) == len(self.fasta), f"Length of fasta file {len(self.fasta)} not equal to split file {len(df_label)}"

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
        print(f"idx: {idx}")

        seq = self.samples[idx]
        print(f"seq: {seq}")
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
        print(data.tolist())
        return data, target



class SPLASHDataLoader(HG38):
    _name_ = "splash"
    #l_output = 0  # need to set this for decoder to work correctly
    #TODO: may add back when doing fine-tuning

    def __init__(self, fasta_file, label_file, tokenizer_name=None, dataset_config_name=None, d_output=None, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, replace_N_token=False, replace_X_token=True,
                 task='next_token_pred', remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.replace_N_token = replace_N_token
        self.replace_X_token = replace_X_token
        self.task = task
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test
        self.d_output = d_output
        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N', "X"], # X is for separating anchor-target pairs
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("**using pretrained AIRI tokenizer**")
            self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            del self.dataset_train
        if hasattr(self, 'dataset_val'):
            del self.dataset_val
        if hasattr(self, 'dataset_test'):
            del self.dataset_test


        # Create all splits: torch datasets
        self.dataset_train = SPLASHDatset(
            split='train',
            fasta_file=self.fasta_file,
            label_file=self.label_file,
            max_length=self.max_length,
            pad_max_length=self.max_length,
            tokenizer=self.tokenizer,
            tokenizer_name=self.tokenizer_name,
            add_eos=self.add_eos,
            task=self.task,
            replace_N_token=self.replace_N_token,
            replace_X_token=self.replace_X_token,
        )

        self.dataset_val = SPLASHDatset(
            split='val',
            fasta_file=self.fasta_file,
            label_file=self.label_file,
            max_length=self.max_length,
            pad_max_length=self.max_length_val,
            tokenizer=self.tokenizer,
            tokenizer_name=self.tokenizer_name,
            add_eos=self.add_eos,
            task=self.task,
            replace_N_token=self.replace_N_token,
            replace_X_token=self.replace_X_token,
        )

        self.dataset_test = SPLASHDatset(
            split='test',
            fasta_file=self.fasta_file,
            label_file=self.label_file,
            max_length=self.max_length,
            pad_max_length=self.max_length_test,
            tokenizer=self.tokenizer,
            tokenizer_name=self.tokenizer_name,
            add_eos=self.add_eos,
            task=self.task,
            replace_N_token=self.replace_N_token,
            replace_X_token=self.replace_X_token,
        )

        return


if __name__ == "__main__":
    # test SPLASHDatset
    fasta_file = "/scratch/users/khoang99/outs/splash_outs/run_07_31_23/RE_CTGCAG_pSpectral_lt_01.fasta"
    # label_file = "data/label.csv"
    label_file = None
    max_length = 1024
    tokenizer_name = "char"
    tokenizer = CharacterTokenizer(characters=['A', 'C', 'G', 'T', 'N', "X"],\
                                     model_max_length=max_length + 2,\
                                     add_special_tokens=False)   
    dataset = SPLASHDatset("train", fasta_file, label_file, max_length,\
                             tokenizer_name=tokenizer_name, tokenizer=tokenizer, task = "classification")
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])


    # # test SPLASHDataLoader
    # fasta_file = "data/hg38.fa"
    # label_file = "data/label.csv"
    # max_length = 1024
    # tokenizer_name = "bpe"
    # batch_size = 32
    # num_workers = 1
    # dataset_config_name = "splash"
    # dataloader = SPLASHDataLoader(fasta_file, label_file, tokenizer_name=tokenizer_name, dataset_config_name=dataset_config_name, max_length=max_length, batch_size=batch_size, num_workers=num_workers)
    # print(len(dataloader.dataset_train))
    # print(len(dataloader.dataset_val))

    