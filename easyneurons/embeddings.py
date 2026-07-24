import numpy as np
from pyparsing import srange
from collections import defaultdict
from pympler import asizeof
import os
import struct
from pathlib import Path

class CoOccurrenceMatrix:
    BYTES_PER_PAIR = 192.0

    def __init__(self):
        pass

    # wighted - czy wyniki mają być ważone od odległości
    # max_bytes_in_ram - ile pamięci max będzie trzymana w ram
    #
    def calculate(self, tokenized_corpus: tuple[tuple[int]],
                  sliding_window: int = 3,
                  weighted: bool = True,
                  max_bytes_in_ram: int = 64 * 1024 * 1024,
                  check_ram_every_n_tokens: int = 20000,
                  shard_directory: str = "./shards_data"
                  ):
        shard_filenames: list[str] = []
        Path(shard_directory).mkdir(parents=True, exist_ok=True)

        buffer: dict[tuple[int,int], float] = defaultdict(float)
        tokens_since_check: int = 0

        max_pairs_estimate = max(1, int(max_bytes_in_ram / self.BYTES_PER_PAIR))

        print(max_pairs_estimate)

        for sentence in tokenized_corpus:
            n_length = len(sentence)
            for position in range(n_length):
                token = sentence[position]

                left = max(0, position - sliding_window)
                right = min(n_length, position + sliding_window + 1)

                for j in range(left, right):
                    if j == position:
                        continue

                    weight = 1.0 / abs(j - position) if weighted else 1.0
                    buffer[token, sentence[j]] += weight

                tokens_since_check += n_length

                if tokens_since_check > check_ram_every_n_tokens:
                    tokens_since_check = 0

                    if len(buffer) > max_pairs_estimate:
                        shard_filenames.append(self._flush_buffer_to_shard(buffer, shard_directory=shard_directory, shard_idx=len(shard_filenames)))
                        buffer.clear()

        if buffer:
            shard_filenames.append(self._flush_buffer_to_shard(buffer, shard_directory=shard_directory, shard_idx=len(shard_filenames)))
            buffer.clear()

        




    @staticmethod
    def _flush_buffer_to_shard(buffer: dict, shard_directory: str, shard_idx: int) -> str:
        filename = os.path.join(shard_directory, f'com_shard_{shard_idx:05d}.bin')
        with open(filename, 'wb') as f:
            for (a, b), value in sorted(buffer.items()):
                f.write(struct.pack("<iid", a, b, value))

        return filename

    def __iter__(self):
        pass

class GloVe():
    pass

if __name__ == '__main__':
    com = CoOccurrenceMatrix()
    com.calculate([[1,2,3,4,5], [5,4,3,2,1]], sliding_window=3, weighted=True, max_bytes_in_ram=180, check_ram_every_n_tokens=-1)
