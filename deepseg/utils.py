# Copyright 2018 luozhouyang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os


def check_src_vocab_file(params):
    """ Check src vocab file, adding special tokens to it.

    Args:
        params: A dict, hyper params
    """
    if not os.path.exists(params['src_vocab']):
        raise ValueError("src_vocab must be provided in this mode!")
    vocab_size = check_vocab_file(
        params['src_vocab'], [params['unk'], params['pad']])
    params['vocab_size'] = vocab_size


def check_tag_vocab_file(params):
    """Check tag vocab file, adding special tokens to it.

    Args:
        params: A dict, hyper params
    """
    if not os.path.exists(params['tag_vocab']):
        raise ValueError("tag_vocab must be provided in this mode!")
    vocab_size = check_vocab_file(params['tag_vocab'], [params['oov_tag']])
    params['num_tags'] = vocab_size


def check_vocab_file(file, special_tokens):
    """Check a vocab file, adding special tokens to it.

    Args:
        file: A constant string, path of the vocab file
        special_tokens: A list of special tokens

    Returns:
        The size of the processed vocab.
    """
    vocabs = set()
    with open(file, mode="rt", encoding="utf8", buffering=8192) as f:
        for line in f:
            line = line.strip("\n").strip()
            if not line:
                continue
            if line in special_tokens:
                continue
            vocabs.add(line)
    vocabs = sorted(vocabs)
    for t in reversed(special_tokens):
        vocabs.insert(0, t)
    with open(file, mode="wt", encoding="utf8", buffering=8192) as f:
        for v in vocabs:
            f.write(v + "\n")
    return len(vocabs)
