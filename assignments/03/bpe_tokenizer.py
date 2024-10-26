from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import sys

def main():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(vocab_size=30000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train([sys.argv[1], sys.argv[2]], trainer)
    tokenizer.save(sys.argv[3])

if __name__ == "__main__":
    main()