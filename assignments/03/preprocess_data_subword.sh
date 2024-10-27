#!/bin/bash
# -*- coding: utf-8 -*-

set -e

# source C:/Users/User/Desktop/MT/atmt311/Scripts/activate

pwd=`dirname "$(readlink -f "$0")"`
src=fr
tgt=en
# data=$pwd/data/$1/
base=$pwd/../..
cd $base

# train BPE tokenizer with subword-nmt library
cat $pwd/data/parallel/preprocessed/train.$tgt $pwd/data/parallel/preprocessed/train.$src | subword-nmt learn-bpe -s 8000 -o $pwd/data/parallel/preprocessed/bpe.codes

for model in parallel copied translated back-translated
do
    # change into base directory to ensure paths are valid
    data=$pwd/data/$model/

    # create preprocessed directory
    mkdir -p $data/preprocessed/

    # normalize and truecase raw data
    cat $data/raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed/train.$src.p
    cat $data/raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed/train.$tgt.p

    # train truecase models
    perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$src --corpus $data/preprocessed/train.$src.p
    perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$tgt --corpus $data/preprocessed/train.$tgt.p

    # apply truecase models to splits
    cat $data/preprocessed/train.$src.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/train.$src 
    cat $data/preprocessed/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/train.$tgt 

    # train BPE tokenizer with tokenizers library
    # cat $data/preprocessed/train.$tgt $data/preprocessed/train.$src | subword-nmt learn-bpe -s 8000 -o $data/preprocessed/bpe.codes
    # python $pwd/bpe_tokenizer.py $data/preprocessed/train.$src $data/preprocessed/train.$tgt $data/preprocessed/vocab.json

    # Apply BPE tokenizer to splits
    for split in train valid test tiny_train
    do 
        for lang in $src $tgt
        do
            # subword-nmt apply-bpe -c $data/preprocessed/bpe.codes < $data/preprocessed/$split.$lang | subword-nmt get-vocab > $data/preprocessed/$split.vocab.$lang
            # subword-nmt apply-bpe -c $data/preprocessed/bpe.codes < $data/preprocessed/$split.$lang | subword-nmt get-vocab > $data/preprocessed/$split.vocab.$lang
            # subword-nmt apply-bpe -c $data/preprocessed/bpe.codes --vocabulary $data/preprocessed/$split.vocab.$lang --vocabulary-threshold 10 < $data/preprocessed/$split.$lang > $data/preprocessed/$split.bpe.$lang
            subword-nmt apply-bpe -c $data/preprocessed/bpe.codes < $data/preprocessed/$split.$lang > $data/preprocessed/$split.bpe.$lang
            # python $pwd/bpe_tokenizer_apply.py $data/preprocessed/vocab.json $data/preprocessed/$split.$lang $data/preprocessed/$split.bpe.$lang
        done
    done

    # preprocess all files for model training
    python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/ --train-prefix $data/preprocessed/train.bpe --valid-prefix $data/preprocessed/valid.bpe --test-prefix $data/preprocessed/test.bpe --tiny-train-prefix $data/preprocessed/tiny_train.bpe --threshold-src 1 --threshold-tgt 1 --num-words-src 3000 --num-words-tgt 3000
done
echo "done!"