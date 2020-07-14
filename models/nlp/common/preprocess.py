"""
Inspiration from https://github.com/google-research/electra/blob/master/build_pretraining_dataset.py

TODO: Parse into documents.
I might accidentally cross document boundaries with this technique.
How many examples should I generate?
Assert we don't generate any examples less than 512

Common mistakes: Can't serialize example? Try VarLenFeature, that means your examples aren't all the same length.

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

import random
from functools import partial

import nlp
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast

from common.datasets import get_electra_dataset

load_from_cache_file = True

print("Loading dataset")
dset = nlp.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
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


def join_documents(batch):
    """ Each document starts with a `= Title =`, and subheadings have two/three equals signs. """


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


def tokenize(batch):
    """ Tokenize via list comprehension in Python. """
    return {"tokens": [tokenizer.tokenize(example) for example in batch["sentences"]]}


def tokenize(batch):
    """ Tokenize via list comprehension in Rust. """
    return {"tokens": tokenizer.tokenize_batch(batch["sentences"])}


dset = dset.select(np.arange(0, 60000))
print("Tokenizing sentences:")
dset = dset.map(tokenize, batched=True, remove_columns=["sentences"])
print("Tokenized sentences:", dset, dset[0])

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
    partial(create_examples, max_length=512),
    batched=True,
    remove_columns=["tokens"],
    load_from_cache_file=load_from_cache_file,
)
print("Created examples:", dset, dset[0])
# WARNING: Some of these examples are shorter than 512 sequence length.
# View with [len(ex["examples"]) for ex in dset]


def batch_ids_from_pretokenized(batch):
    exs = batch["examples"]
    ret_val = tokenizer(
        [list(ex) for ex in exs],
        is_pretokenized=True,
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    return ret_val


cache_file = "/Users/nieljare/Desktop/wikitext2-encoded.cache"
print("Padding, truncating, and encoding examples into ids")
dset = dset.map(
    batch_ids_from_pretokenized,
    batched=True,
    remove_columns=["examples"],
    cache_file_name=cache_file,
    load_from_cache_file=load_from_cache_file,
)
print("Padded, truncated, and encoded examples into ids:", dset, dset[0])
# dset = nlp.Dataset.from_file(cache_file)

filename = "/tmp/dset.tfrecord"
dset.export(filename)

### Now read in a TFRecord to ensure exporting happened correctly ###

max_seq_length = 512
name_to_features = {
    "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),  # corresponds to input_ids
    "token_type_ids": tf.io.FixedLenFeature(
        [max_seq_length], tf.int64
    ),  # corresponds to token_type_ids
    "attention_mask": tf.io.FixedLenFeature(
        [max_seq_length], tf.int64,
    ),  # corresponds to attention_mask
}

tfds = get_electra_dataset(
    filenames=[filename], max_seq_length=512, per_gpu_batch_size=4, shard=False
)
for batch in tfds.take(1):
    print(batch)
