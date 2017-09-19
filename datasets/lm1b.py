from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import os
import tarfile


from six.moves import xrange

import tensorflow as tf
from tefla.dataset.textdataset import TextDataset
from tefla.dataset.tokenizer import InvertibleTokenizer


class Lm1b32k(TextDataset):
    """A language model on the 1B words corpus."""

    def __init__(self):
        self.tokenizer = InvertibleTokenizer()
        self.EOS = 1
        super(Lm1b32k, self).__init__(self)

    @property
    def is_character_level(self):
        return False

    @property
    def has_inputs(self):
        return True

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def num_shards(self):
        return 100

    @property
    def vocab_name(self):
        return "vocab.lm1b.en"

    @property
    def use_subword_tokenizer(self):
        return True

    @property
    def targeted_vocab_size(self):
        return 2**15  # 32768

    @property
    def use_train_shards_for_dev(self):
        return True

    def generator(self, tmp_dir, train, characters=False):
        """Generator for lm1b sentences.

        Args:
          tmp_dir: a string.
          train: a boolean.
          characters: a boolean

        Yields:
          A dictionary {"inputs": [0], "targets": [<subword ids>]}
        """
        _maybe_download_corpus(tmp_dir)
        original_vocab = _original_vocab(tmp_dir)
        files = (_train_data_filenames(tmp_dir) if train
                 else [_dev_data_filename(tmp_dir)])
        if characters:
            encoder = text_encoder.ByteTextEncoder()
        else:
            encoder = _get_or_build_subword_text_encoder(
                tmp_dir, self.tokenizer)
        for filepath in files:
            tf.logging.info("filepath = %s", filepath)
            for line in tf.gfile.Open(filepath):
                tokens = encoder.encode(
                    _replace_oov(original_vocab, text_encoder.native_to_unicode(line)))
                tokens.append(self.EOS)
                yield {"inputs": [0], "targets": tokens}


def _original_vocab(tmp_dir):
    """Returns a set containing the original vocabulary.

    This is important for comparing with published results.

    Args:
      tmp_dir: directory containing dataset.

    Returns:
      a set of strings
    """
    vocab_url = ("http://download.tensorflow.org/models/LM_LSTM_CNN/"
                 "vocab-2016-09-10.txt")
    vocab_filename = os.path.basename(vocab_url + ".en")
    vocab_filepath = os.path.join(tmp_dir, vocab_filename)
    if not os.path.exists(vocab_filepath):
        maybe_download(tmp_dir, vocab_filename, vocab_url)
    return set(
        [text_encoder.native_to_unicode(l.strip()) for l in
         tf.gfile.Open(vocab_filepath)])


def maybe_download(directory, filename, url):
    """Download filename from url unless it's already in directory.

    Args:
      directory: path to the directory that will be used.
      filename: name of the file to download to (do nothing if it already exists).
      url: URL to download from.

    Returns:
      The path to the downloaded file.
    """
    if not tf.gfile.Exists(directory):
        tf.logging.info("Creating directory %s" % directory)
        os.mkdir(directory)
    filepath = os.path.join(directory, filename)
    if not tf.gfile.Exists(filepath):
        tf.logging.info("Downloading %s to %s" % (url, filepath))
        inprogress_filepath = filepath + ".incomplete"
        inprogress_filepath, _ = urllib.urlretrieve(
            url, inprogress_filepath, reporthook=download_report_hook)
        # Print newline to clear the carriage return from the download progress
        print()
        tf.gfile.Rename(inprogress_filepath, filepath)
        statinfo = os.stat(filepath)
        tf.logging.info("Succesfully downloaded %s, %s bytes." % (filename,
                                                                  statinfo.st_size))
    else:
        tf.logging.info("Not downloading, file already found: %s" % filepath)
    return filepath


def _replace_oov(original_vocab, line):
    """Replace out-of-vocab words with "UNK".

    This maintains compatibility with published results.

    Args:
      original_vocab: a set of strings (The standard vocabulary for the dataset)
      line: a unicode string - a space-delimited sequence of words.

    Returns:
      a unicode string - a space-delimited sequence of words.
    """
    return u" ".join(
        [word if word in original_vocab else u"UNK" for word in line.split()])


def _train_data_filenames(tmp_dir):
    return [os.path.join(
        tmp_dir,
        "1-billion-word-language-modeling-benchmark-r13output",
        "training-monolingual.tokenized.shuffled",
        "news.en-%05d-of-00100" % i) for i in xrange(1, 100)]


def _dev_data_filename(tmp_dir):
    return os.path.join(
        tmp_dir,
        "1-billion-word-language-modeling-benchmark-r13output",
        "heldout-monolingual.tokenized.shuffled",
        "news.en.heldout-00000-of-00050")


def _maybe_download_corpus(tmp_dir):
    """Download and unpack the corpus.

    Args:
      tmp_dir: directory containing dataset.
    """
    corpus_url = ("http://www.statmt.org/lm-benchmark/"
                  "1-billion-word-language-modeling-benchmark-r13output.tar.gz")
    corpus_filename = os.path.basename(corpus_url)
    corpus_filepath = os.path.join(tmp_dir, corpus_filename)
    if not os.path.exists(corpus_filepath):
        maybe_download(tmp_dir, corpus_filename, corpus_url)
        with tarfile.open(corpus_filepath, "r:gz") as corpus_tar:
            corpus_tar.extractall(tmp_dir)


def _get_or_build_subword_text_encoder(tmp_dir, tokenizer_obj):
    """Builds a SubwordTextEncoder based on the corpus.

    Args:
      tmp_dir: directory containing dataset.
    Returns:
      a SubwordTextEncoder.
    """
    filepath = os.path.join(tmp_dir, "lm1b_32k.subword_text_encoder")
    if tf.gfile.Exists(filepath):
        return text_encoder.SubwordTextEncoder(filepath)
    _maybe_download_corpus(tmp_dir)
    original_vocab = _original_vocab(tmp_dir)
    token_counts = defaultdict(int)
    line_count = 0
    max_lines = 63000
    for line in tf.gfile.Open(_train_data_filenames(tmp_dir)[0]):
        tokens = tokenizer_obj.encode(
            _replace_oov(original_vocab, text_encoder.native_to_unicode(line)))
        for tok in tokens:
            token_counts[tok] += 1
        line_count += 1
        if line_count >= max_lines:
            break
    ret = text_encoder.SubwordTextEncoder()
    ret.build_from_token_counts(token_counts, min_count=5)
    ret.store_to_file(filepath)
    return ret
