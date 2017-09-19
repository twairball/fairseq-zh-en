from __future__ import print_function
import argparse
import os
import jieba
import nltk

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# init
jieba.initialize()

def _preprocess_sgm(line, is_sgm):
    """Preprocessing to strip tags in SGM files."""
    if not is_sgm:
        return line
    # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
    if line.startswith("<srcset") or line.startswith("</srcset"):
        return ""
    if line.startswith("<refset") or line.startswith("</refset"):
        return ""
    if line.startswith("<doc") or line.startswith("</doc"):
        return ""
    if line.startswith("<p>") or line.startswith("</p>"):
        return ""
    # Strip <seg> tags.
    line = line.strip()
    if line.startswith("<seg") and line.endswith("</seg>"):
        i = line.index(">")
        return line[i+1:-6]  # Strip first <seg ...> and last </seg>.


def tokenize(filepath, lower_case=True, delim=' '):
    filename = os.path.basename(filepath)
    is_sgm = filename.endswith(".sgm")
    is_zh = filename.endswith(".zh") or filename.endswith(".zh.sgm")

    tokenized = ''
    f = open(filepath, 'r')
    for i, line in enumerate(f):
        
        if i % 2000 == 0:
            _tokenizer_name = "jieba" if is_zh else "nltk.word_tokenize" 
            logger.info("     [%d] %s: %s" % (i, _tokenizer_name, line))

        # strip sgm tags if any
        _line = _preprocess_sgm(line, is_sgm)
        # tokenize
        _tok = jieba.cut(_line.rstrip('\r\n')) if is_zh else nltk.word_tokenize(_line)
        _tokenized = delim.join(_tok)
        # lowercase. ignore if chinese. 
        _tokenized = _tokenized.lower() if lower_case and not is_zh else _tokenized
        # append
        tokenized += _tokenized
        tokenized += "\n"
    f.close()
    return tokenized


parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='input filepath')
parser.add_argument('--output', required=True, help='output filepath')
parser.add_argument('--delim', required=False, default=" ", help='delimiter, default=" "')
parser.add_argument('--lowercase', required=False, default=True, help='lower case, default=True. Ignored for .zh')

if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    tokenized = tokenize(opt.input, delim=opt.delim)
    fo = open(opt.output, 'w')
    fo.write(tokenized)
    fo.close()    
