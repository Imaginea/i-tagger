#!/bin/bash 

if [ "$#" -ne 1 ]; then
    echo "Enter file path."
    exit 1
fi
if [ ! -f $1 ]; then
    echo "File not found!"
    exit 1
fi
echo "Started conversion" 
sed 's/ \+/,/g' $1 > temp_file
sed 's/^$/WORD,DEPENDENCY,IOB_DEPENDENCY,Entity/g' temp_file > test.csv 
csplit --prefix=test -b "%d.csv" -z test.csv '/^WORD/' {*}
rm test0.csv temp_file
