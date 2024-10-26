from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm
import sys
import os

def merge_files(file1, file2, output_file):
    with open(output_file, 'w') as outfile:
        for fname in [file1, file2]:
            with open(fname, 'r') as infile:
                outfile.write(infile.read() + "\n")

def main():
    folder_path = os.path.dirname(sys.argv[1])
    merged_file_path = os.path.join(folder_path, 'train.all') 
    merge_files(sys.argv[1], sys.argv[2],merged_file_path)
    spm.SentencePieceTrainer.train(input=merged_file_path, model_prefix=os.path.join(folder_path, 'bpe_model'), vocab_size=10000, model_type='bpe')
    # tokenizer = Tokenizer(BPE())
    # trainer = BpeTrainer(vocab_size=30000, min_frequency=1)
    # tokenizer.pre_tokenizer = Whitespace()
    # tokenizer.train([sys.argv[1], sys.argv[2]], trainer)
    # tokenizer.save(sys.argv[3])

if __name__ == "__main__":
    main()