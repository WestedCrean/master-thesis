#!/bin/zsh

if [ ! -d "../data/numbers" ]; then
    mkdir ../data/numbers
fi

# for loop from 0 to 9 inclusive
for i in ({0..9}); do
    #cp -r ../data/all_characters/$i ../data/numbers/$i
    echo "../data/all_characters_raw/$i"
done

