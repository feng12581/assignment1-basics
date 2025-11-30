import os
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
import time
import json
from collections import defaultdict

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def tasks(chunk: str):
    dict = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for pre_token in re.finditer(PAT, chunk):
        pre_token_utf8 = pre_token.group().encode("utf-8")
        dict[tuple(bytes([byte]) for byte in pre_token_utf8)] += 1
    return dict


def pre_tokenization(input_file: str, special_tokens : list[str]):
    special_token = special_tokens[0]
    assert special_token == "<|endoftext|>"
    with open(input_file, "rb") as f:
        num_processes = os.cpu_count() * 2
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            text = re.split(re.escape(special_token), f.read(end - start).decode("utf-8", errors="ignore"))
            for item in text:
                chunks.append(item)

        with Pool(num_processes) as pool:
            result = pool.map(tasks, chunks)
        
        merged_dict : dict[tuple[bytes, ...], int] = defaultdict(int)
        #pretoken_pos : dict[int, tuple[bytes, ...]] = {}

        for d in result:
            for k, v in d.items():
                # print(k)
                merged_dict[k] += v
        
        table_pre_token_pos : dict[int, tuple[tuple[bytes, ...], int]] = {}
        # table_merged_pre_token : dict[int, tuple[bytes, ...]] = {}
        for index, (pre_token, n) in enumerate(merged_dict.items()):
            table_pre_token_pos[index] = (pre_token, n)
            # table_merged_pre_token[index] = pre_token
  
        return (merged_dict, table_pre_token_pos)

