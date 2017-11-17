# Use 1 archive per speaker
# Take 5 samples per archive
# Take 500 samples per language
per_lang=500
filename='voxforge_set1.csv'
rm -f $filename
grep "\(,1\|,2\|,3\|,4\|,5\)$" voxforge_samples.log | grep German | awk -F, '{print $1","$2}' | head -n $per_lang >> $filename
grep "\(,1\|,2\|,3\|,4\|,5\)$" voxforge_samples.log | grep English | awk -F, '{print $1","$2}' | head -n $per_lang >> $filename
grep "\(,1\|,2\|,3\|,4\|,5\)$" voxforge_samples.log | grep Spanish | awk -F, '{print $1","$2}' | head -n $per_lang >> $filename
grep "\(,1\|,2\|,3\|,4\|,5\)$" voxforge_samples.log | grep Italian | awk -F, '{print $1","$2}' | head -n $per_lang >> $filename
grep "\(,1\|,2\|,3\|,4\|,5\)$" voxforge_samples.log | grep Portuguese | awk -F, '{print $1","$2}' | head -n $per_lang >> $filename
shuf $filename -o $filename
wc -l $filename
head $filename
