from tokenizers import Tokenizer
import sentencepiece as spm
import sys
import os

def add_subword_suffix(tokens):
    # Encode the text into pieces
    processed_tokens = []
    for token in tokens:
        # Add `@@` if the token does not start with the `▁` marker (not a standalone word)
        if not token.startswith('▁'):
            processed_tokens.append(token + '@@')
        else:
            # Remove `▁` for standalone words
            processed_tokens.append(token[1:])
    return ' '.join(processed_tokens).replace("▁", " ")

def main():
    # print(sys.argv[1])
    # tokenizer = Tokenizer.from_file(sys.argv[1])
    # Load the trained model
    folder_path = os.path.dirname(sys.argv[1])
    sp = spm.SentencePieceProcessor(model_file=os.path.join(folder_path, 'bpe_model.model'))
    with open(sys.argv[2], "r", encoding="utf-8") as f_in, open(sys.argv[3], "w", encoding="utf-8") as f_out:
        for line in f_in:
            tokens = sp.encode(line, out_type=str)
            processed_text = add_subword_suffix(tokens)
            f_out.write(str(processed_text) + "\n")

if __name__ == "__main__":
    main()