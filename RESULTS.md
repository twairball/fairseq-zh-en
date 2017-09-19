# Results


## wmt17 en-zh

### Bi-LSTM

````
| Test with beam=1: BLEU4 = 64.59, 69.1/63.7/63.1/62.7 (BP=1.000, ratio=0.995, sys_len=545557, ref_len=543024)
| Test with beam=5: BLEU4 = 64.15, 75.0/70.2/69.8/69.6 (BP=0.902, ratio=1.103, sys_len=492180, ref_len=543024)
| Test with beam=10: BLEU4 = 64.38, 74.8/70.2/69.8/69.5 (BP=0.906, ratio=1.099, sys_len=494193, ref_len=543024)
| Test with beam=20: BLEU4 = 63.76, 73.8/69.1/68.6/68.3 (BP=0.912, ratio=1.092, sys_len=497325, ref_len=543024)
````

### fconv
words/s: 9750

````
| Test with beam=1: BLEU4 = 63.49, 74.2/67.5/66.3/65.6 (BP=0.929, ratio=1.073, sys_len=505902, r
ef_len=543024)
| Test with beam=5: BLEU4 = 62.22, 79.4/73.8/73.0/72.6 (BP=0.834, ratio=1.182, sys_len=459550, r
ef_len=543024)
| Test with beam=10: BLEU4 = 62.52, 79.9/74.6/73.9/73.6 (BP=0.828, ratio=1.189, sys_len=456857, 
ref_len=543024)
| Test with beam=20: BLEU4 = 62.74, 79.9/74.8/74.1/73.8 (BP=0.830, ratio=1.187, sys_len=457641, 
ref_len=543024)
````

### fconv dropout 0.5
words/s: 9750
epochs: 35
train loss: 1.8
train ppl: 3.58

````
$ fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model fconv -nenclayer 4 -nlayer 3 -dropout 0.5 -optim nag -lr 0.25 -clip 0.1 \
    -momentum 0.99 -timeavg -bptt 0 -savedir trainings/fconv_dropout0.5


| Test with beam=1: BLEU4 = 49.43, 63.8/53.9/51.6/50.3 (BP=0.904, ratio=1.101, sys_len=493298, ref_len=543024)
| Test with beam=5: BLEU4 = 47.87, 72.1/63.4/61.7/60.7 (BP=0.744, ratio=1.296, sys_len=419111, ref_len=543024)
| Test with beam=10: BLEU4 = 47.60, 74.3/66.2/64.8/63.9 (BP=0.709, ratio=1.344, sys_len=403961, ref_len=543024)
| Test with beam=20: BLEU4 = 47.51, 74.5/66.8/65.5/64.7 (BP=0.701, ratio=1.355, sys_len=400716, ref_len=543024)

````

### fconv enc_layer 7
words/s: 8876
epochs: 33
train loss: 0.53
train ppl: 1.44

````
$ mkdir -p trainings/fconv_nenclayer7
$ fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model fconv -nenclayer 7 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
    -momentum 0.99 -timeavg -bptt 0 -savedir trainings/fconv_nenclayer7

| Test with beam=1: BLEU4 = 66.40, 75.8/69.7/68.6/67.9 (BP=0.943, ratio=1.059, sys_len=512739, ref_len=543024)
| Test with beam=5: BLEU4 = 65.52, 80.9/75.8/75.0/74.6 (BP=0.856, ratio=1.156, sys_len=469933, ref_len=543024)
| Test with beam=10: BLEU4 = 65.80, 81.4/76.6/75.9/75.5 (BP=0.851, ratio=1.161, sys_len=467757, ref_len=543024)
| Test with beam=20: BLEU4 = 65.96, 81.4/76.7/76.0/75.7 (BP=0.852, ratio=1.160, sys_len=468072, ref_len=543024)

# you actually have to implement the solution
# <unk> 实际上 必须 实施 解决办法 。
````

### fconv layer 5
words/s: 8519
epochs: 28
train loss: 0.56
train ppl: 1.48

````
$ mkdir -p trainings/fconv_nlayer5
$ fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model fconv -nenclayer 4 -nlayer 5 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
    -momentum 0.99 -timeavg -bptt 0 -savedir trainings/fconv_nlayer5

| Test with beam=1: BLEU4 = 65.65, 75.5/69.4/68.2/67.6 (BP=0.937, ratio=1.065, sys_len=509668, ref_len=543024)
| Test with beam=5: BLEU4 = 64.71, 80.1/74.9/74.1/73.7 (BP=0.855, ratio=1.156, sys_len=469630, ref_len=543024)
| Test with beam=10: BLEU4 = 64.91, 80.7/75.8/75.1/74.8 (BP=0.848, ratio=1.165, sys_len=466256, ref_len=543024)
| Test with beam=20: BLEU4 = 64.98, 80.6/75.9/75.3/75.0 (BP=0.848, ratio=1.165, sys_len=465969, ref_len=543024)

# you actually have to implement the solution
# <unk> 实际上 必须 实施 解决办法 。
````

### ConvEnc
words/s: 7835
epochs: 47

````
| Test with beam=1: BLEU4 = 50.91, 56.4/49.9/49.1/48.6 (BP=1.000, ratio=0.896, sys_len=606309, ref_len=543024)
| Test with beam=5: BLEU4 = 56.71, 65.4/59.4/58.9/58.7 (BP=0.937, ratio=1.065, sys_len=509942, ref_len=543024)
| Test with beam=10: BLEU4 = 56.83, 63.0/57.4/56.9/56.6 (BP=0.973, ratio=1.028, sys_len=528438, ref_len=543024)
| Test with beam=20: BLEU4 = 53.66, 57.9/52.9/52.3/51.7 (BP=1.000, ratio=0.950, sys_len=571676, ref_len=543024)
````