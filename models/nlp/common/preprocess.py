"""
Inspiration from https://github.com/google-research/electra/blob/master/build_pretraining_dataset.py
25 seconds for WikiText-2 (2M tokens, 84k sentences)
40 minutes for WikiText-103 (103M tokens, 4.1M sentences)
?? minutes for Wikipedia (2500M tokens, 31M sentences)

The steps are:
1) Download data
2) Filter empty lines (112k it/s)
3) Replace newlines with space (121k it/s)
4) Split on periods into sentences (66k it/s)
5) Pre-tokenize sentences (12k it/s)
6) Create examples (24k it/s)
7) Convert example tokens into ids (0.15k it/s) -> because of casting ndarray to list?
8) Export to TFRecords

We can tokenize Wikipedia in 6 minutes, and create examples in 3.
Takes 90 minutes to convert into ids.


TODO: Parse into documents.

To directly inspect a TFRecord without knowing the spec:
tfds = tf.data.TFRecordDataset(filenames=[filename])
for batch in tfds.take(1):
    example_proto = tf.train.Example.FromString(batch.numpy())

To attempt loading in a VarLenFeature to see if you didn't serialize everything the same length:
features = {
    "input_ids": tf.io.VarLenFeature(tf.int64),
    "token_type_ids": tf.io.VarLenFeature(tf.int64),
    "attention_mask": tf.io.VarLenFeature(tf.int64),
}
"""

import argparse
import os
import random
import time
from functools import partial
from typing import List

import nlp
import tensorflow as tf
from transformers import BertTokenizerFast

from common.datasets import get_electra_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument(
    "--dataset", choices=["wikitext-2", "wikitext-103", "wikipedia", "bookcorpus", "wikibooks"]
)
parser.add_argument("--cache_folder", default=None)
parser.add_argument("--shards", type=int, default=1)
parser.add_argument("--tfrecord_folder", default="/tmp")
parser.add_argument("--skip_load_from_cache_file", action="store_true")
args = parser.parse_args()

load_from_cache_file = not args.skip_load_from_cache_file

start_time = time.perf_counter()

print(f"Loading dataset: {args.dataset}")
if args.dataset.startswith("wikitext"):
    dset = nlp.load_dataset("wikitext", f"{args.dataset}-raw-v1", split="train")
elif args.dataset == "wikipedia":
    dset = nlp.load_dataset("wikipedia", "20200501.en", split="train")
    dset.drop(columns=["title"])
elif args.dataset == "bookcorpus":
    dset = nlp.load_dataset("bookcorpus", split="train")
elif args.dataset == "wikibooks":
    dset_wikipedia = nlp.load_dataset("wikipedia", "20200501.en", split="train")
    dset_books = nlp.load_dataset("bookcorpus", split="train")
    dset = nlp.Dataset.from_concat([dset_wikipedia, dset_books])
else:
    assert False
print("Loaded dataset:", dset, dset[0])
print("Filtering empty lines")
dset = dset.filter(lambda ex: len(ex["text"]) > 0)
print("Filtered empty lines:", dset, dset[0])
print("Replacing newlines with space")
dset = dset.map(
    lambda batch: {"text": [text.strip().replace("\n", " ") for text in batch["text"]]},
    batched=True,
)
print("Replaced newlines with space:", dset, dset[0])


def split_into_sentences(batch):
    """ Split into sentences using the '.' separator. Not perfect, converts

    Senjō no Valkyria 3 : Unrecorded Chronicles (
    Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the
    Battlefield 3 ) , commonly referred to as Valkyria
    Chronicles III outside Japan , is a tactical role
    @-@ playing video game developed by Sega and
    Media.Vision for the PlayStation Portable .

    into three sentences when it really is one. But works pretty well.
    """
    sentences = []
    for ex in batch["text"]:
        batch_sentences = [sentence + "." for sentence in ex.split(".")]
        batch_sentences = batch_sentences[:-1]
        sentences.extend(batch_sentences)
    return {"sentences": sentences}


print("Splitting into sentences")
dset = dset.map(
    split_into_sentences,
    batched=True,
    remove_columns=["text"],
    load_from_cache_file=load_from_cache_file,
)
print("Split into sentences:", dset, dset[0])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# def tokenize(batch):
#     """ Tokenize via list comprehension in Python. """
#     return {"tokens": [tokenizer.tokenize(example) for example in batch["sentences"]]}


def tokenize(batch):
    """ Tokenize via list comprehension in Rust. """
    encodings: List["Encoding"] = tokenizer._tokenizer.encode_batch(batch["sentences"])
    tokens: List[str] = [encoding.tokens for encoding in encodings]
    return {"tokens": tokens}


# dset = dset.select(np.arange(0, 60000))
print("Pre-tokenizing sentences:")
dset = dset.map(tokenize, batched=True, remove_columns=["sentences"])
print("Pre-tokenized sentences:", dset, dset[0])

# def tokens_to_ids(first_segment, second_segment):
#     sequence = ["[CLS]"] + first_segment + ["[SEP]"] + second_segment + ["[SEP]"]
#     return tokenizer.convert_tokens_to_ids(sequence)


def create_examples(batch, max_length):
    """Creates a pre-training example from the current list of sentences."""
    target_length = max_length
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
        first_segment_target_length = 100000
    else:
        # -3 due to not yet having [CLS]/[SEP] tokens in the input text
        first_segment_target_length = (target_length - 3) // 2

    first_segment, second_segment = [], []
    examples = []
    for sentence in batch["tokens"]:
        # the sentence goes to the first segment if (1) the first segment is
        # empty, (2) the sentence doesn't put the first segment over length or
        # (3) 50% of the time when it does put the first segment over length
        if (
            len(first_segment) == 0
            or len(first_segment) + len(sentence) < first_segment_target_length
            or (
                len(second_segment) == 0
                and len(first_segment) < first_segment_target_length
                and random.random() < 0.5
            )
        ):
            first_segment += list(sentence)
        else:
            second_segment += list(sentence)
            if len(first_segment) + len(second_segment) >= target_length:
                # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
                first_segment = first_segment[: max_length - 2]
                second_segment = second_segment[: max(0, max_length - len(first_segment) - 3)]
                example = ["[CLS]"] + first_segment + ["[SEP]"] + second_segment + ["[SEP]"]
                examples.append(example)
                first_segment, second_segment = [], []

                if random.random() < 0.05:
                    target_length = random.randint(5, max_length)
                else:
                    target_length = max_length

    # This last one may be a little short, but it's necessary to always return something from the function
    # for the function inspection that only passes two sentences.
    examples.append(["[CLS]"] + first_segment + ["[SEP]"] + second_segment + ["[SEP]"])

    return {"examples": examples}


print("Creating examples")
dset = dset.map(
    partial(create_examples, max_length=args.max_seq_length),
    batched=True,
    remove_columns=["tokens"],
    load_from_cache_file=load_from_cache_file,
)
print("Created examples:", dset, dset[0])
# WARNING: Some of these examples are shorter than 512 sequence length.
# View with [len(ex["examples"]) for ex in dset]


# This method is very slow (0.15 it/s, so 0.15k examples/sec
# Improvement tracked in https://github.com/huggingface/transformers/issues/5729
def batch_ids_from_pretokenized(batch):
    exs = batch["examples"]
    ret_val = tokenizer(
        [list(ex) for ex in exs],
        is_pretokenized=True,
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_length,
    )
    return ret_val


print("Padding, truncating, and encoding examples into ids")
dset = dset.map(
    batch_ids_from_pretokenized,
    batched=True,
    remove_columns=["examples"],
    # cache_file_name=args.cache_file,
    load_from_cache_file=load_from_cache_file,
)
print("Padded, truncated, and encoded examples into ids:", dset, dset[0])
# dset = nlp.Dataset.from_file(cache_file)

tfrecord_files = [
    os.path.join(args.tfrecord_folder, f"{args.dataset}_shard_{i}.tfrecord")
    for i in range(args.shards)
]
for i in range(args.shards):
    dset_shard = dset.shard(num_shards=args.shards, index=i)
    dset_shard.export(tfrecord_files[i])

### Now read in a TFRecord to ensure exporting happened correctly ###

name_to_features = {
    "input_ids": tf.io.FixedLenFeature([args.max_seq_length], tf.int64),  # corresponds to input_ids
    "token_type_ids": tf.io.FixedLenFeature(
        [args.max_seq_length], tf.int64
    ),  # corresponds to token_type_ids
    "attention_mask": tf.io.FixedLenFeature(
        [args.max_seq_length], tf.int64,
    ),  # corresponds to attention_mask
}

tfds = get_electra_dataset(
    filenames=tfrecord_files, max_seq_length=args.max_seq_length, per_gpu_batch_size=4, shard=False,
)
for batch in tfds.take(1):
    print(batch)

elapsed = time.perf_counter() - start_time
print(f"Total processing time: {elapsed:.3f} seconds")