import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from summarizer import pack_sentences, split_sentences


def test_split_sentences_basic():
    text = "Hello world.  How are you? Fine!\nThanks."
    assert split_sentences(text) == ["Hello world.", "How are you?", "Fine!", "Thanks."]


def test_split_sentences_empty():
    assert split_sentences("   \n ") == []


def test_pack_respects_limit():
    sents = ["a", "b", "c", "d", "e"]
    chunks = pack_sentences(sents, [4, 4, 4, 4, 4], max_tokens=10)
    assert chunks == ["a b", "c d", "e"]


def test_pack_long_sentence_gets_own_chunk():
    chunks = pack_sentences(["short", "huge", "short2"], [2, 50, 2], max_tokens=10)
    assert chunks == ["short", "huge", "short2"]


def test_pack_keeps_all_sentences_in_order():
    sents = [f"s{i}" for i in range(20)]
    chunks = pack_sentences(sents, [3] * 20, max_tokens=10)
    assert " ".join(chunks).split() == sents
