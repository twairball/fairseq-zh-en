#!/bin/sh

TEXT=data/wmt17_en_zh
DATADIR=data-bin/wmt17_en_zh
TRAIN=trainings/wmt17_en_zh

# Fully convolutional sequence-to-sequence model
mkdir -p $TRAIN/fconv
fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model fconv -nenclayer 4 -nlayer 3 -batchsize 16 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
    -momentum 0.99 -timeavg -bptt 0 -savedir $TRAIN/fconv

# Standard bi-directional LSTM model
mkdir -p $TRAIN/blstm
fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model blstm -nhid 512 -dropout 0.2 -dropout_hid 0 -optim adam -lr 0.0003125 \
    -savedir $TRAIN/blstm

