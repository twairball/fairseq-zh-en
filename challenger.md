TEXT=data/challenger
DATADIR=data-bin/challenger.en.zh
TRAINDIR=trainings/challenger_fconv

# Tokenize Chinese 
python -m jieba $TEXT/sample_train.zh -d" " > $TEXT/data.tok.zh

# Tokenize English
python preprocess/nltk_tokenize.py --input $TEXT/sample_train.en --output $TEXT/data.tok.en

# make train, val, test datasets
python preprocess/make_dataset.py --en $TEXT/data.tok.en --zh $TEXT/data.tok.zh \
    --output_dir=$TEXT/

fairseq preprocess -sourcelang en -targetlang zh \
    -trainpref $TEXT/train -validpref $TEXT/valid -testpref $TEXT/test \
    -thresholdsrc 3 -thresholdtgt 3 -destdir $DATADIR


# Training

echo "starting training, DATADIR=" $DATADIR

# Fully convolutional sequence-to-sequence model
mkdir -p $TRAINDIR
fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model fconv -nenclayer 4 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
    -momentum 0.99 -timeavg -bptt 0 -savedir $TRAINDIR
