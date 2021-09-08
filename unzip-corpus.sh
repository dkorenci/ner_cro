#!/usr/bin/env bash
# this is in separate script since '>' (probably) cannot be escaped in spacy's yaml
archive=$1
fileloc=$2
outfile=$3
unzip -p $1 $2 > $3