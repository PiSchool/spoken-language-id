#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Provide input and output csv files."
    exit
fi

# Grep regex for limiting counts per speaker
limit_count_re="\(,1\|,2\|,3\|,4\|,5\|,6\|,7\|,8\|,9\|,10\|,11\|,12\|,13\|,14\|,15\|,16\|,17\|,18\|,19\|,20\)$"

# Use 1 archive per speaker
# Take 5 samples per archive
# Take 500 samples per language
per_lang=2070
in_file=$1
out_file=$2
rm -f $out_file
grep "$limit_count_re" $in_file | grep German | head -n $per_lang >> $out_file
grep "$limit_count_re" $in_file | grep French | head -n $per_lang >> $out_file
grep "$limit_count_re" $in_file | grep English | head -n $per_lang >> $out_file
grep "$limit_count_re" $in_file | grep Spanish | head -n $per_lang >> $out_file
grep "$limit_count_re" $in_file | grep Italian | head -n $per_lang >> $out_file
grep "$limit_count_re" $in_file | grep Portuguese | head -n $per_lang >> $out_file

# Shuffled so that user files are grouped together, but languages are mixed
sort -R -t, -k 3.1,3.12 $out_file -o $out_file

# Only keep the two first columns (filename, language)
tmp_file=$(mktemp /tmp/trainset.XXXXXX)
awk -F, '{print $1","$2}' $out_file > $tmp_file
mv $tmp_file $out_file

wc -l $out_file
head $out_file
