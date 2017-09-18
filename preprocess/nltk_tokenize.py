from __future__ import print_function
import argparse
import nltk

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='input filepath')
parser.add_argument('--output', required=True, help='output filepath')
parser.add_argument('--delim', required=False, default=" ", help='delimiter, default=" "')

"""
Tokenizer with nltk/stanford 
"""
def tokenize(filepath, delim=' '):
    f = open(filepath, 'r')
    tokenized = ''
    for line in f:
        tokenized += delim.join(nltk.word_tokenize(line.lower()))
        tokenized += "\n"
    f.close()
    return tokenized

if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    tokenized = tokenize(opt.input, delim=opt.delim)
    fo = open(opt.output, 'w')
    fo.write(tokenized)
    fo.close()    
