#!/bin/bash

# check if phcd.rar is present in src folder, if yes then don't download it
if [ ! -f "phcd.rar" ]; then
    echo "phcd.rar not found in src folder"
    # download the dataset
    wget -O phcd.rar https://cs.pollub.pl/phcd/phcd.rar 
    echo "Dataset downloaded."
else
    echo "phcd.rar already present"
fi

mkdir ../data

# copy to ../data/
mv phcd.rar ../data/phcd.rar

echo "Unpacking ..."

# unpack the archive
cd ../data && unrar x phcd.rar && unrar x phsf.rar

echo "Unpacked. Cleaning up ..."

# copy image folders to their final directory
mkdir all_characters
mv znaki/png/* all_characters/.
rm -rf *.py *.pdf *.rar ocr_files inwokacja znaki

echo "Done!"
