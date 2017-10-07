#!/bin/sh

DATADIR=data-bin/wmt17_en_zh
TRAIN=trainings/wmt17_en_zh
OUTPUT=tmp/wmt17_en_zh/fconv_test

echo "optimizing fconv for decoding"
fairseq optimize-fconv -input_model $TRAIN/fconv/model_best.th7 -output_model $TRAIN/fconv/model_best_opt.th7

echo "decoding to ${OUTPUT}"
fairseq generate -path $TRAIN/fconv/model_best_opt.th7 -datadir $DATADIR \
     -beam 10 -nbest 2 -dataset test -sourcelang en -targetlang zh | tee $OUTPUT.tmp

# TODO: decode subword BPE
cat $OUTPUT.tmp | sed -r 's/(@@ )|(@@ ?$)//g' > $OUTPUT.out