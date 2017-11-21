#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Provide 2 input and 1 output csv files."
    exit
fi

# Grep regex for limiting counts per speaker
limit_count_re="\(,1\|,2\|,3\|,4\|,5\|,6\|,7\|,8\|,9\|,10\|,11\|,12\|,13\|,14\|,15\|,16\|,17\|,18\|,19\|,20\)$"

# Use this many VoxForge samples per language
voxforge_per_lang=1520

# Remove output file if exists
out_file=$3
rm -f $out_file

voxforge_in_file=$1
grep "$limit_count_re" $voxforge_in_file | grep German | head -n $voxforge_per_lang >> $out_file
grep "$limit_count_re" $voxforge_in_file | grep French | head -n $voxforge_per_lang >> $out_file
grep "$limit_count_re" $voxforge_in_file | grep English | head -n $voxforge_per_lang >> $out_file
grep "$limit_count_re" $voxforge_in_file | grep Spanish | head -n $voxforge_per_lang >> $out_file
grep "$limit_count_re" $voxforge_in_file | grep Italian | head -n $voxforge_per_lang >> $out_file
grep "$limit_count_re" $voxforge_in_file | grep Portuguese | head -n $voxforge_per_lang >> $out_file


# Use this many Audio Lingua samples per language
audiolingua_per_lang=210

audiolingua_in_file=$2
grep "$limit_count_re" $audiolingua_in_file | grep German | head -n $audiolingua_per_lang >> $out_file
grep "$limit_count_re" $audiolingua_in_file | grep French | head -n $audiolingua_per_lang >> $out_file
grep "$limit_count_re" $audiolingua_in_file | grep English | head -n $audiolingua_per_lang >> $out_file
grep "$limit_count_re" $audiolingua_in_file | grep Spanish | head -n $audiolingua_per_lang >> $out_file
grep "$limit_count_re" $audiolingua_in_file | grep Italian | head -n $audiolingua_per_lang >> $out_file
grep "$limit_count_re" $audiolingua_in_file | grep Portuguese | head -n $audiolingua_per_lang >> $out_file

# Shuffled so that user files are grouped together, but languages are mixed
sort -R -t"," -k3,3 $out_file -o $out_file

# Only keep the two first columns (filename, language)
tmp_file=$(mktemp /tmp/trainset.XXXXXX)
awk -F, '{print $1","$2}' $out_file > $tmp_file
mv $tmp_file $out_file

wc -l $out_file
head $out_file
