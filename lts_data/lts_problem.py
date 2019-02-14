# coding=utf-8
""" Problem definition for translation from Chinese to English."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

#from tensor2tensor.data_generators.wmt import WMTProblem
from tensor2tensor.data_generators.translate import TranslateProblem
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

import json


# Chinese to English translation datasets.
LOCATION_OF_DATA = '/var/storage/shared/sdrgvc/xuta/t-hasu/lts/lts_data/'
_LTS_TRAIN_DATASETS = [
    LOCATION_OF_DATA+'lts.s2s.train',
]


_LTS_DEV_DATASETS = [
    LOCATION_OF_DATA+'lts.s2s.test',
]

LTS_VOCAB_FILES = [
    LOCATION_OF_DATA+'lts.l.vocab',
    LOCATION_OF_DATA+'lts.s.vocab',
]


def bi_vocabs_token2id_generator(data_path, source_token_vocab, target_token_vocab, eos=None):
    """Generator for sequence-to-sequence tasks that uses tokens.

    This generator assumes the files at source_path and target_path have
    the same number of lines and yields dictionaries of "inputs" and "targets"
    where inputs are token ids from the " "-split source (and target, resp.) lines
    converted to integers using the token_map.

    Args:
      source_path: path to the file with source sentences.
      target_path: path to the file with target sentences.
      source_token_vocab: text_encoder.TextEncoder object.
      target_token_vocab: text_encoder.TextEncoder object.
      eos: integer to append at the end of each sequence (default: None).

    Yields:
      A dictionary {"inputs": source-line, "targets": target-line} where
      the lines are integer lists converted from tokens in the file lines.
    """
    eos_list = [] if eos is None else [eos]
    with tf.gfile.GFile(data_path, mode="r") as data_file:

            data = data_file.readline()
            while data:
                source, target, teacher = data.strip().split('\t')
                source_ints = source_token_vocab.encode(source.strip()) + eos_list
                target_ints = target_token_vocab.encode(target.strip()) + eos_list
                #print(source_ints,target_ints)
                teacher_ints_ = json.loads(teacher)[:len(target_ints)]
                teacher_ints=[]
                for ti in teacher_ints_:
                    teacher_ints+=ti[1:]
                #print("lengths",len(source_ints),len(target_ints),len(teacher_ints))
                #print({"inputs": source_ints, "targets": target_ints, "teacher": teacher_ints})
                yield {"inputs": source_ints, "targets": target_ints, "teacher": teacher_ints}
                data = data_file.readline()


@registry.register_problem
class LTS(TranslateProblem):
    """Problem spec for WMT17 Zh-En translation."""


    @property
    def target_vocab_size(self):
        return 28440 # subtract for compensation

    @property
    def num_shards(self):
        return 1

    @property
    def source_vocab_name(self):
        return "vocab.src.%d" % self.targeted_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.tgt.%d" % self.targeted_vocab_size

    @property
    def input_space_id(self):
        return problem.SpaceID.ZH_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK
    
    
    # Pre-process two vocabularies and build a generator.
    def generator(self, data_dir, tmp_dir, train):
        # Load source vocabulary.
        tf.logging.info("Loading and processing source vocabulary for %s from:" % ("training" if train else "validation"))
        print('    ' + LTS_VOCAB_FILES[0] + ' ... ', end='')
        sys.stdout.flush()
        with open(LTS_VOCAB_FILES[0], 'r', encoding='utf8') as f:
            vocab_src_list = f.read().splitlines()
        print('Done')
        
        # Load target vocabulary.
        tf.logging.info("Loading and processing target vocabulary for %s from:" % ("training" if train else "validation"))
        print('    ' + LTS_VOCAB_FILES[1] + ' ... ', end='')
        sys.stdout.flush()
        with open(LTS_VOCAB_FILES[1], 'r', encoding='utf8') as f:
            vocab_trg_list = f.read().splitlines()
        print('Done')
        
        # Truncate the vocabulary depending on the given size (strip the reserved tokens).
        vocab_src_list = vocab_src_list[3:]
        vocab_trg_list = vocab_trg_list[3:]
    
        # Insert the <UNK>.
        vocab_src_list.insert(0, "<UNK>")
        vocab_trg_list.insert(0, "<UNK>")
    
        # Auto-insert the reserved tokens as: <pad>=0 <EOS>=1 and <UNK>=2.
        source_vocab = text_encoder.TokenTextEncoder(vocab_filename=None, vocab_list=vocab_src_list,
                                                     replace_oov="<UNK>", num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS)
        target_vocab = text_encoder.TokenTextEncoder(vocab_filename=None, vocab_list=vocab_trg_list,
                                                     replace_oov="<UNK>", num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS)
        
        # Select the path: train or dev (small train).
        datapath = _LTS_TRAIN_DATASETS if train else _LTS_DEV_DATASETS
        
        # Build a generator.
        return bi_vocabs_token2id_generator(datapath[0], source_vocab, target_vocab, text_encoder.EOS_ID)
    
    
    # Build bi-vocabs feature encoders for decoding.
    def feature_encoders(self, data_dir):
        # Load source vocabulary.
        tf.logging.info("Loading and processing source vocabulary from: %s" % LTS_VOCAB_FILES[0])
        with open(LTS_VOCAB_FILES[0], 'r', encoding="utf-8") as f:
            vocab_src_list = f.read().splitlines()
        tf.logging.info("Done")
        
        # Load target vocabulary.
        tf.logging.info("Loading and processing target vocabulary from: %s" % LTS_VOCAB_FILES[1])
        with open(LTS_VOCAB_FILES[1], 'r', encoding="utf-8") as f:
            vocab_trg_list = f.read().splitlines()
        tf.logging.info("Done")
    
        # Truncate the vocabulary depending on the given size (strip the reserved tokens).
        vocab_src_list = vocab_src_list[3:]
        vocab_trg_list = vocab_trg_list[3:]
    
        # Insert the <UNK>.
        vocab_src_list.insert(0, "<UNK>")
        vocab_trg_list.insert(0, "<UNK>")
    
        # Auto-insert the reserved tokens as: <pad>=0 <EOS>=1 and <UNK>=2.
        source_encoder = text_encoder.TokenTextEncoder(vocab_filename=None, vocab_list=vocab_src_list,
                                                       replace_oov="<UNK>", 
                                                        num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS)
        target_encoder = text_encoder.TokenTextEncoder(vocab_filename=None, vocab_list=vocab_trg_list,
                                                       replace_oov="<UNK>",
                                                       num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS)        
        
        return {"inputs": source_encoder, "targets": target_encoder}




