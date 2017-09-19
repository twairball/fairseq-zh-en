
# Preprocess

````
# download, unzip, tokenize datasets
# We use dataset_config dictionary -- see wmt.py for more. 
$ python prepare/challenger.py

$ TEXT=data/challenger_nmt
$ DATADIR=data-bin/challenger_nmt
$ fairseq preprocess -sourcelang en -targetlang zh \
    -trainpref $TEXT/train -validpref $TEXT/valid -testpref $TEXT/valid \
    -thresholdsrc 3 -thresholdtgt 3 -destdir $DATADIR
````

# Training

````
$ DATADIR=data-bin/challenger_nmt

# Standard bi-directional LSTM model
$ mkdir -p trainings/challenger_nmt/blstm
$ fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model blstm -nhid 512 -dropout 0.2 -dropout_hid 0 -optim adam -lr 0.0003125 \
    -savedir trainings/challenger_nmt/blstm

# Fully convolutional sequence-to-sequence model
$ mkdir -p trainings/challenger_nmt/fconv
$ fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model fconv -nenclayer 4 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
    -momentum 0.99 -timeavg -bptt 0 -savedir trainings/challenger_nmt/fconv

````