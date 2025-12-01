import os
from time import sleep
from .pretokenization_example import pre_tokenization
from collections import defaultdict
import pdb
from typing import Iterable, Iterator
import regex as re
from multiprocessing import Pool

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        if special_tokens is not None:
            for token in special_tokens:
                if token not in vocab.keys():
                    length = len(vocab)
                    vocab[length] = token.encode("utf-8")
        
        self.reverse_vocab = dict((v, k) for k, v in vocab.items())
        
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        pass

    def encode(self, text: str) -> list[int]:
        # pdb.set_trace()
        token_ids: list[int] = []
        if self.special_tokens is not None:
            PAT = "(" + "|".join([re.escape(item) for item in self.special_tokens]) + ")"
            segments = self.function(re.split(PAT, text))
        else:
            segments = text

        num_process = 1

        for segment in segments:
            token_ids += self.generate_token_id(segment)
        # with Pool(num_process) as pool:
        #     result = pool.map(self.generate_token_id, segments)

        # for tokens in result:
        #     token_ids += tokens
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError
    
    def decode(self, ids: list[int]) -> str:
        unibytes: bytes = bytes()
        for id in ids:
            unibytes += self.vocab[id]
        return unibytes.decode("utf-8")

    def function(self, text_split: list[str]) -> list[str]:
        result: list[str] = []
        special = str()
        for i in range(len(text_split)):
            s = text_split[i]
            if s == "":
                if i == len(text_split) - 1:
                    assert len(special) > 0
                    result.append(special)
                    break
                else:
                    continue
            if s in self.special_tokens:
                special += s
            else:
                if len(special) > 0:
                    result.append(special)
                    result.append(s)
                    special = ""
                else:
                    result.append(s)
        return result

    def generate_token_id(self, segment: str) -> list[int]:
        flag = True
        token_ids: list[int] = []
        if self.special_tokens is not None:
            for token in self.special_tokens:
                if token in segment:
                    flag = False
                    break

        if flag:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for pre_token in re.finditer(PAT, segment):
                pre_token_utf8 = pre_token.group().encode("utf-8")
                token_ids += self.apply_merge(pre_token_utf8)
                
        else:
            pos: list[tuple[int, int]] = []
            left = 0
            for i in range(len(segment)):
                right = i + 1
                if segment[left:right] in self.special_tokens:
                    pos.append((left, right))
                    left = right
                
            merged_pos: list[tuple[int, int]] = pos
            done = False
            while not done:
                length = len(merged_pos)
                if length == 1:
                    done = True
                else:
                    i = 0
                    while i < length - 1:
                        left = merged_pos[i][0]
                        right = merged_pos[i+1][1]
                        if segment[left:right] in self.special_tokens:
                            head = merged_pos[0:i]
                            middle = [(left, right)]
                            tail = merged_pos[i+2:]
                            merged_pos = head + middle + tail
                            break
                        else:
                            if i == length - 2:
                                done = True
                            i += 1
            
            for (left, right) in merged_pos:
                special_token = segment[left:right]
                assert special_token in self.special_tokens
                token_ids.append(self.reverse_vocab[special_token.encode("utf-8")])
        return token_ids

    
    def apply_merge(self, segment_utf8: bytes) -> list[int]:
        # pdb.set_trace()
        merged_segment: list[bytes] = [segment_utf8[i:i+1] for i in range(len(segment_utf8))]
        done = False
        while not done:
            length = len(merged_segment)
            if length == 1:
                done = True
            else:
                i = 0
                # pdb.set_trace()
                merge_point = -1
                pos = len(self.merges)
                while i < length - 1:
                    bp = (merged_segment[i], merged_segment[i+1])
                    if bp in self.merges:
                        index = self.merges.index(bp)
                        if index < pos:
                            pos = index
                            merge_point = i
                    i += 1
                if merge_point < 0:
                    break
                else:
                    head = merged_segment[0:merge_point]
                    middle = [merged_segment[merge_point] + merged_segment[merge_point+1]]
                    tail = merged_segment[merge_point+2:]
                    merged_segment = head + middle + tail


        token_ids: list[int] = []

        for token in merged_segment:
            token_ids.append(self.reverse_vocab[token])
        return token_ids



def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def get_all_max_keys(dictionary : defaultdict[tuple[bytes, bytes], int]):
    max_value = max(dictionary.values())
    return [key for key, value in dictionary.items() if value == max_value]

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocabulary : dict[int, bytes] = {}
    merges : list[tuple[bytes, bytes]] = []
    for index, item in enumerate(special_tokens):
        vocabulary[index] = item.encode("utf-8")
    d = gpt2_bytes_to_unicode()
    for index, (key, value) in enumerate(d.items()):
        vocabulary[index+1] = key.to_bytes()
    token_nums = len(vocabulary)
    assert token_nums == 256 + len(special_tokens)
    assert token_nums < vocab_size

    pre_tokens, table_pre_token_pos = pre_tokenization(input_path, special_tokens)
    byte_pairs_frequency : dict[tuple[bytes, bytes], int] = defaultdict(int)
    byte_pairs_source : dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    # byte_pairs_per_pretoken: dict[tuple[bytes, ...], tuple[tuple[bytes, bytes]]] = {}
    # merged_pre_token : dict[tuple[bytes, ...], tuple[bytes, ...]] = {}
    table_bps_per_pretoken : dict[tuple[bytes, ...], tuple[tuple[bytes, bytes]]] = {}
    for pos in range(len(table_pre_token_pos)):
        (pre_token, n) = table_pre_token_pos[pos]
        if len(pre_token) == 1:
            continue
        bps : list[tuple[bytes, bytes]] = []
        for i in range(len(pre_token) - 1):
            bp = (pre_token[i], pre_token[i+1])
            byte_pairs_frequency[bp] += n
            byte_pairs_source[bp].add(pos)
            bps.append(bp)
        table_bps_per_pretoken[pre_token] = tuple(bps)

    while token_nums < vocab_size:
        max_frequency_bp = max(get_all_max_keys(byte_pairs_frequency))
        merges.append(max_frequency_bp)
        new_token = max_frequency_bp[0] + max_frequency_bp[1]
        vocabulary[token_nums] = new_token
        token_nums += 1

        for pos in byte_pairs_source[max_frequency_bp]:
            (pre_token, n) = table_pre_token_pos[pos]
            bps_before_merge = table_bps_per_pretoken[pre_token]

            if max_frequency_bp not in bps_before_merge:
                continue

            length = len(bps_before_merge)
            pre_token_after_merge : list[tuple[bytes, bytes]] = []

            index = 0
            while index < length:
                bp = bps_before_merge[index]
                if length == 1:
                    assert bp == max_frequency_bp
                    # pre_token_after_merge.append(new_token)
                    index += 1
                    continue
                if index < length - 1:
                    bp_after = bps_before_merge[index + 1]
                    if bp == max_frequency_bp:
                        if bp_after == max_frequency_bp:
                            if index + 2 < length and bps_before_merge[index + 2] == max_frequency_bp:
                                byte_pairs_frequency[(new_token, new_token)] += n
                                byte_pairs_source[(new_token, new_token)].add(pos)
                                pre_token_after_merge.append((new_token, new_token))
                                index += 2
                                continue
                            else:
                                byte_pairs_frequency[(new_token, bp_after[1])] += n
                                byte_pairs_source[(new_token, bp_after[1])].add(pos)
                                pre_token_after_merge.append((new_token, bp_after[1]))
                                index += 2
                                continue
                        else:
                            byte_pairs_frequency[bp_after] -= n
                            byte_pairs_frequency[(new_token, bp_after[1])] += n
                            byte_pairs_source[(new_token, bp_after[1])].add(pos)
                            pre_token_after_merge.append((new_token, bp_after[1]))
                            index += 2
                            continue
                    else:
                        if bp_after == max_frequency_bp:
                            byte_pairs_frequency[(bp[0], new_token)] += n
                            byte_pairs_frequency[bp] -= n
                            byte_pairs_source[(bp[0], new_token)].add(pos)
                            pre_token_after_merge.append((bp[0], new_token))
                            index += 1
                            continue
                        else:
                            pre_token_after_merge.append(bp)
                            index += 1
                            continue
                else:
                    if bp != max_frequency_bp:
                        pre_token_after_merge.append(bp)
                    index += 1
                    
                
            table_bps_per_pretoken[pre_token] = tuple(pre_token_after_merge)

        # pdb.set_trace()
        byte_pairs_frequency.pop(max_frequency_bp)

    return (vocabulary, merges)
