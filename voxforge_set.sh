#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Provide input and output csv files."
    exit
fi

# Use 1 archive per speaker
# Take 5 samples per archive
# Take 500 samples per language
per_lang=500
in_file=$1
out_file=$2
rm -f $out_file
grep "\(,1\|,2\|,3\|,4\|,5\)$" $in_file | grep German | awk -F, '{print $1","$2}' | head -n $per_lang >> $out_file
grep "\(,1\|,2\|,3\|,4\|,5\)$" $in_file | grep English | awk -F, '{print $1","$2}' | head -n $per_lang >> $out_file
grep "\(,1\|,2\|,3\|,4\|,5\)$" $in_file | grep Spanish | awk -F, '{print $1","$2}' | head -n $per_lang >> $out_file
grep "\(,1\|,2\|,3\|,4\|,5\)$" $in_file | grep Italian | awk -F, '{print $1","$2}' | head -n $per_lang >> $out_file
grep "\(,1\|,2\|,3\|,4\|,5\)$" $in_file | grep Portuguese | awk -F, '{print $1","$2}' | head -n $per_lang >> $out_file
shuf $out_file -o $out_file
wc -l $out_file
head $out_file
