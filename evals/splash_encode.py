import torch 

import argparse
import os
import sys
import yaml 
from tqdm import tqdm
import json 
from pathlib import Path
from pyfaidx import Fasta
from src.models.sequence.dna_embedding import DNAEmbeddingModel

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

try:
    from tokenizers import Tokenizer  
except:
    pass

# https://github.com/openai/gpt-2/issues/131#issuecomment-492786058
# def preprocess(text):
#     text = text.replace("“", '"')
#     text = text.replace("”", '"')
#     return '\n'+text.strip()


class SPLASHEncoder:
    "Encoder inference for HG38 sequences"
    def __init__(self, model_cfg, ckpt_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.model, self.tokenizer = self.load_model(model_cfg, ckpt_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def encode(self, seqs):
            
        results = []

        # sample code to loop thru each sample and tokenize first (char level)
        for seq in tqdm(seqs):
            
            if isinstance(self.tokenizer, Tokenizer):
                tokenized_seq = self.tokenizer.encode(seq).ids
            else:
                tokenized_seq = self.tokenizer.encode(seq)
            
            # can accept a batch, shape [B, seq_len, hidden_dim]
            logits, __ = self.model(torch.tensor([tokenized_seq]).to(device=self.device))

            # Using head, so just have logits
            results.append(logits)

        return results
        
            
    def load_model(self, model_cfg, ckpt_path):
        config = yaml.load(open(model_cfg, 'r'), Loader=yaml.FullLoader)
        model = DNAEmbeddingModel(**config['model'])
        
        state_dict = torch.load(ckpt_path, map_location='cpu')

        # loads model from ddp by removing prexix to single if necessary
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )

        model_state_dict = state_dict["state_dict"]

        # need to remove torchmetrics. to remove keys, need to convert to list first
        for key in list(model_state_dict.keys()):
            if "torchmetrics" in key:
                model_state_dict.pop(key)

        model.load_state_dict(state_dict["state_dict"])

        # setup tokenizer
        if config['dataset']['tokenizer_name'] == 'char':
            print("**Using Char-level tokenizer**")

            # add to vocab
            tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N', 'X'],
                model_max_length=self.max_seq_len + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
            print(tokenizer._vocab_str_to_int)
        else:
            raise NotImplementedError("You need to provide a custom tokenizer!")

        return model, tokenizer
        
        
if __name__ == "__main__":
    
    default_config_path = Path(__file__).parent.parent.absolute() / "configs" / "experiment" / "splash" / "splash_pretrain.yaml"

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_cfg",
        default=default_config_path,
    )
    
    parser.add_argument(
        "--ckpt_path",
        default=f"",
        help="Path to model state dict checkpoint"
    )

    parser.add_argument(
        "--input_fasta",
        default=f"",
        help="Path to fasta file used to generate embeddings"
    )

    parser.add_argument(
        "--output_dir",
        default=f"",
        help="Path to directory to save embeddings"
    )
        
    args = parser.parse_args()
        
    task = SPLASHEncoder(args.model_cfg, args.ckpt_path, max_seq_len=1024)

    # read in fasta file
    if not os.path.exists(args.input_fasta):
        print(f"File {args.input_fasta} does not exist")
        sys.exit(1)
    # check if fasta ends in .fasta or .fa
    if not args.input_fasta.endswith(".fasta") and not args.input_fasta.endswith(".fa"):
        print(f"File {args.input_fasta} is not a fasta file")
        sys.exit(1)
    # read in fasta file
    fasta = Fasta(args.input_fasta)
    seqs = [list(str(seq)) for seq in fasta]

    batch_size = 32
    # encode sequences in batches
    embeddings = []
    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i:i+batch_size]
        embeddings.extend(task.encode(batch_seqs))
    
    print("Num seqs", len(embeddings))
    print("Indivisual embedding shape", embeddings[0].shape)

    # get average, last, and first embeddings
    stacked_embeddings = torch.stack(embeddings).squeeze()
    avg_embeddings = torch.mean(stacked_embeddings, dim=1)
    last_embeddings = stacked_embeddings[:, -1, :]
    first_embeddings = stacked_embeddings[:, 0, :]
    print(avg_embeddings.shape)
    print(last_embeddings.shape)
    print(first_embeddings.shape)
    # save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(avg_embeddings, os.path.join(args.output_dir, "avg_embeddings.pt"))
    torch.save(last_embeddings, os.path.join(args.output_dir, "last_embeddings.pt"))
    torch.save(first_embeddings, os.path.join(args.output_dir, "first_embeddings.pt"))
    

    # sample sequence, can pass a list of seqs (themselves a list of chars)
    # example1 = list("ACGTACGTACACGTACGTACGTACGTACGTACGTGTACGTACGTACGTACGTACGTACGTACGTACGTACGT")
    # example2 = list("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT")

    # seqs = [example1, example2]

    # embeddings = task.encode(seqs)
    # print(embeddings)
    # print(len(embeddings))
    # print(embeddings[0].shape)
    # print(logits[0].logits.shape)

    # # check if can read in fasta file
    # if not os.path.exists(args.input_fasta):
    #     print(f"File {args.input_fasta} does not exist")
    #     sys.exit(1)
    # # check if fasta ends in .fasta or .fa
    # if not args.input_fasta.endswith(".fasta") and not args.input_fasta.endswith(".fa"):
    #     print(f"File {args.input_fasta} is not a fasta file")
    #     sys.exit(1)
    # # read in fasta file
    # fasta = Fasta(args.input_fasta)
    # seqs = [str(seq) for seq in fasta]

    # logits = task.encode(seqs)
    # print(logits[0].logits.shape)



    