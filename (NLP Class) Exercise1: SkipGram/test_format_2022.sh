#!/bin/bash

# brief: script to test format of deliverables of Exercise #1
# This is part of an assignment of the NLP module (DSBA, CentraleSupelec & ESSEC). More info: https://sites.google.com/view/cs-dsba-nlp/home

# USAGE: ./test_script.sh filename.tar.gz

tar zxfv $1


SIMLEX_URL=https://www.cl.cam.ac.uk/~fh295/SimLex-999.zip
#BWC_URL=https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
BWC_URL=files.europe.naverlabs.com/index.php/files/getfile/42ad94d7c71563b88ee50fc80cfdbd87

TESTDATA=simlex.csv
TRAINDATA=train.txt
RESULTFILE=results.txt
REPORT=report.pdf
NLINES=1000 # number of lines, play around with increasing numbers


# download training text data
if [ -f 1-billion-word-language-modeling-benchmark-r13output.tar.gz ]
then
    echo "INFO: Billion word corpus already downloaded, skipping downloading"
else
	wget $BWC_URL
	mv 42ad94d7c71563b88ee50fc80cfdbd87 1-billion-word-language-modeling-benchmark-r13output.tar.gz
	tar zxfv 1-billion-word-language-modeling-benchmark-r13output.tar.gz
fi

echo "INFO: creating training data"
head -n $NLINES 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00024-of-00100 > $TRAINDATA



# download word similarities
echo "INFO: donwloading word similarity data"
if [ -f $TESTDATA ]
then
	echo "INFO: simlex data already downloaded, skipping"
else
	wget $SIMLEX_URL
	unzip SimLex-999.zip
	sed '1 s/SimLex999/similarity/' SimLex-999/SimLex-999.txt > $TESTDATA
fi

if [ -f $REPORT ]
then
	echo "PASS: report found"
else
	echo "FAIL: missing report"
fi

echo "INFO: running training"
python skipGram.py --model mymodel.model --text $TRAINDATA

echo "INFO: running testing"
python skipGram.py --test --model mymodel.model --text $TESTDATA > $RESULTFILE


echo "INFO: checking format"
echo "INFO: checking number of lines"
# check number of lines is correct
check_lines=$(wc -l $RESULTFILE | cut -f1 -d' ')
if [ "$check_lines" -ne 999 ]; 
then 
	echo "FAIL: should have 999 lines, not $check_lines"
else
	echo "PASS"
fi


# check each line is a number
echo "INFO: checking format of each line"
re="^-?[0-9]*\.?[0-9]*$"
miss=0
for oneline in $(cat $RESULTFILE)
do
	if ! [[ "$oneline" =~ $re ]]; then
		echo "ERROR, line $oneline should be a number"
		miss=$(($miss +1))
	fi
done

if [[ $miss -eq 0 ]];
then
	echo PASS
fi


echo "INFO: cleaning up"
rm $RESULTFILE


