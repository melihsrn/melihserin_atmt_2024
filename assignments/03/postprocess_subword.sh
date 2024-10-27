infile=$1
outfile=$2
lang=$3

sed -r 's/(@@ )|(@@ ?$)//g' $infile > $outfile
cat $outfile | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile
