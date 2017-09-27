from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import gzip
import io
import os
import random
import tarfile

import nltk
import jieba

# Dependency imports

import requests
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import six.moves.urllib_request as urllib  # Imports urllib on Python2, urllib.request on Python3

import tokenizer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Prepare datasets.
heavily borrowed from tensor2tensor's generator_utils.py

dataset_config: {"url": "http://my.dataset.url", "source": "data.en", "target": "data.zh"}
"""
def prepare_dataset(data_dir, tmp_dir, dataset_config, tokenize=True, merge_blanks=True):
    """ download, unzip and copy files to data_dir if necessary """
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    def download_dataset():
        url = dataset_config["url"]
        filename = os.path.basename(url)
        read_type = "r:gz" if "tgz" in filename else "r"

        compressed_file = maybe_download(tmp_dir, filename, url)
        with tarfile.open(compressed_file, read_type) as corpus_tar:
            logger.info("extracting %s to %s" % (compressed_file, tmp_dir))
            corpus_tar.extractall(tmp_dir)

    def get_tmp_file(lang_file):
        tmp_filepath = os.path.join(tmp_dir, lang_file)
        if os.path.isfile(tmp_filepath):
            logger.info("Found file: %s" % data_filepath)
        else:
            # download dataset, if it doesn't exist
            download_dataset()
        return tmp_filepath
    
    for _file in ["source", "target"]:
        _tmp = dataset_config[_file]
        _data = dataset_config["data_%s" % _file]

        # skip if data file exists. 
        data_filepath = os.path.join(data_dir, _data)
        if os.path.isfile(data_filepath):
            logger.info("Found file: %s" % data_filepath)
            continue

        # get tmp file
        tmp_filepath = os.path.join(tmp_dir, _tmp)
        if not os.path.isfile(tmp_filepath):
            logger.info("tmp file: %s not found, downloading..." % tmp_filepath)
            download_dataset()
        
        if tokenize:
            logger.info("tokenizing: %s" % tmp_filepath)
            tokenized = tokenizer.tokenize_file(tmp_filepath)
            logger.info("...done. writing to: %s" % data_filepath)
            f = open(data_filepath, 'w')
            f.write(tokenized)
            f.close()
        else:
            logger.info("tokenize=False, copying to %s" % data_filepath)
            os.rename(tmp_filepath, data_filepath)

    # merge blanks
    if merge_blanks:
        logger.info("\n%s\n%s" % ("=" * 30, "merging blanks..."))
        src = os.path.join(data_dir, dataset_config["data_source"])
        targ = os.path.join(data_dir, dataset_config["data_target"])
        merge_blanks_and_write(src, targ)

def merge_blanks_and_write(src, targ):
    src_lines, targ_lines = _merge_blanks(src, targ, verbose=True)

    logger.info("writing to: %s" % src)
    with open(src, 'w') as f:
        for l in src_lines:            
            f.write(l + "\n")

    logger.info("writing to: %s" % targ)
    with open(targ, 'w') as f:
        for l in targ_lines:
            f.write(l + "\n")
    
def _merge_blanks(src, targ, verbose=False):
    """Read parallel corpus 2 lines at a time. 
    Merge both sentences if only either source or target has blank 2nd line. 
    If both have blank 2nd lines, then ignore. 
    
    Returns tuple (src_lines, targ_lines), arrays of strings sentences. 
    """
    merges_done = [] # array of indices of rows merged
    sub = None # replace sentence after merge
    with open(src, 'rb') as src_file, open(targ, 'rb') as targ_file: 
        src_lines = src_file.readlines()
        targ_lines = targ_file.readlines()
        
        print("src: %d, targ: %d" % (len(src_lines), len(targ_lines)))
        print("=" * 30)
        for i in range(0, len(src_lines) - 1):
            s = src_lines[i].decode().rstrip()
            s_next = src_lines[i+1].decode().rstrip()
            
            t = targ_lines[i].decode().rstrip()
            t_next = targ_lines[i+1].decode().rstrip()
            
            
            if t == '.':
                t = '' 
            if t_next == '.':
                t_next = ''
                
            if (len(s_next) == 0) and (len(t_next) > 0):
                targ_lines[i] = "%s %s" % (t, t_next) # assume it has punctuation
                targ_lines[i+1] = b''
                src_lines[i] = s if len(s) > 0 else sub
                
                merges_done.append(i)
                if verbose: 
                    print("t [%d] src: %s\n      targ: %s" % (i, src_lines[i], targ_lines[i]))
                    print()
                
            elif (len(s_next) > 0) and (len(t_next) == 0):
                src_lines[i] = "%s %s" % (s, s_next) # assume it has punctuation
                src_lines[i+1] = b''
                targ_lines[i] = t if len(t) > 0 else sub
                
                merges_done.append(i)
                if verbose:
                    print("s [%d] src: %s\n      targ: %s" % (i, src_lines[i], targ_lines[i]))
                    print()
            elif (len(s) == 0) and (len(t) == 0):
                # both blank -- remove
                merges_done.append(i)
            else:
                src_lines[i] = s if len(s) > 0 else sub
                targ_lines[i] = t if len(t) > 0 else sub
                
        # handle last line
        s_last = src_lines[-1].decode().strip()
        t_last = targ_lines[-1].decode().strip()
        if (len(s_last) == 0) and (len(t_last) == 0):
            merges_done.append(len(src_lines) - 1)
        else:
            src_lines[-1] = s_last
            targ_lines[-1] = t_last
            
    # remove empty sentences
    for m in reversed(merges_done):
        del src_lines[m]
        del targ_lines[m]
    
    print("merges done: %d" % len(merges_done))
    return (src_lines, targ_lines)
                
                

def download_report_hook(count, block_size, total_size):
    """Report hook for download progress.

    Args:
    count: current block number
    block_size: block size
    total_size: total size
    """
    percent = int(count * block_size * 100 / total_size)
    print("\r%d%%" % percent + " completed", end="\r")

def maybe_download(directory, filename, url):
    """Download filename from url unless it's already in directory.

    Args:
    directory: path to the directory that will be used.
    filename: name of the file to download to (do nothing if it already exists).
    url: URL to download from.

    Returns:
    The path to the downloaded file.
    """
    if not os.path.exists(directory):
        logger.info("Creating directory %s" % directory)
        os.mkdir(directory)
    filepath = os.path.join(directory, filename)

    if not os.path.isfile(filepath):
        logger.info("Downloading %s to %s" % (url, filepath))
        inprogress_filepath = filepath + ".incomplete"
        inprogress_filepath, _ = urllib.urlretrieve(
            url, inprogress_filepath, reporthook=download_report_hook)
        # Print newline to clear the carriage return from the download progress
        print()
        os.rename(inprogress_filepath, filepath)
        statinfo = os.stat(filepath)
        logger.info("Succesfully downloaded %s, %s bytes." % (filename,
                                                              statinfo.st_size))
    else:
        logger.info("Not downloading, file already found: %s" % filepath)
    return filepath


def gunzip_file(gz_path, new_path):
    """Unzips from gz_path into new_path.

    Args:
    gz_path: path to the zipped file.
    new_path: path to where the file will be unzipped.
    """
    if os.path.exists(new_path):
        logger.info("File %s already exists, skipping unpacking" % new_path)
    return
    logger.info("Unpacking %s to %s" % (gz_path, new_path))
    with gzip.open(gz_path, "rb") as gz_file:
        with io.open(new_path, "wb") as new_file:
            for line in gz_file:
                new_file.write(line)
