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
logger = logging.getLogger()
logger.setLevel(logging.INFO)

"""
Prepare datasets.
heavily borrowed from tensor2tensor's generator_utils.py

dataset_config: {"url": "http://my.dataset.url", "source": "data.en", "target": "data.zh"}
"""
def prepare_dataset(data_dir, tmp_dir, dataset_config):
    """ download, unzip and copy files to data_dir if necessary """
        
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
        
        # tokenize
        logger.info("tokenizing: %s" % tmp_filepath)
        tokenized = tokenizer.tokenize(tmp_filepath)
        logger.info("...done. writing to: %s" % data_filepath)
        f = open(data_filepath, 'w')
        f.write(tokenized)
        f.close()

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
