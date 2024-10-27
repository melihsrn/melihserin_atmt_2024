infile=$1
outfile=$2
lang=$3

<<<<<<< HEAD
cat $infile | sed -r 's/(@@ )|(@@ ?$)//g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile
=======
cat $infile | sed -r 's/(@@ )|(@@ ?$)//g' |perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile
>>>>>>> 4e9ffb7aa4977c3f2d84a188aa62862e8ebc14ee
