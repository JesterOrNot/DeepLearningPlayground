#!/bin/bash

kaggle competitions download -c dogs-vs-cats &> /dev/null
mkdir -p data
unzip -o -d data dogs-vs-cats.zip &> /dev/null
rm dogs-vs-cats.zip
cd data
unzip -o train.zip &> /dev/null
rm *.zip
cd ..
