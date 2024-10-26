from tokenizers import Tokenizer
import sys
import os

def main():
    print(sys.argv[1])
    tokenizer = Tokenizer.from_file(sys.argv[1])
    with open(sys.argv[2], "r", encoding="utf-8") as f_in, open(sys.argv[3], "w", encoding="utf-8") as f_out:
        for line in f_in:
            encoded = tokenizer.encode(line.strip())
            f_out.write(" ".join(map(str, encoded.tokens)) + "\n")

if __name__ == "__main__":
    main()