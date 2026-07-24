import numpy as np
from pyparsing import srange
from collections import defaultdict
from pympler import asizeof
import os
import struct
from pathlib import Path
import heapq

class CoOccurrenceMatrix:
    BYTES_PER_PAIR = 192.0
    RECORD_FMT = "<iid"
    RECORD_SIZE = struct.calcsize(RECORD_FMT)

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

        out_path = os.path.join(shard_directory, 'merged_cooccurrence.bin')
        shards = [self._read_shard(f) for f in shard_filenames]
        with open(out_path, 'wb') as out:
            current_key, current_value = None, 0.0

            for key, value in heapq.merge(*shards, key=lambda kv: kv[0]):
                if key == current_key:
                    current_value += value

                else:
                    if current_key is not None:
                        out.write(struct.pack(self.RECORD_FMT, current_key[0], current_key[1], current_value))

                    current_key, current_value = key, value

            if current_key is not None:
                out.write(struct.pack(self.RECORD_FMT, current_key[0], current_key[1], current_value))

        for f in shard_filenames:
            os.remove(f)

        return out_path



    def _flush_buffer_to_shard(self, buffer: dict, shard_directory: str, shard_idx: int) -> str:
        filename = os.path.join(shard_directory, f'com_shard_{shard_idx:05d}.bin')
        with open(filename, 'wb') as f:
            for (a, b), value in sorted(buffer.items()):
                f.write(struct.pack(self.RECORD_FMT, a, b, value))

        return filename

    def _read_shard(self, filename: str, io_buffer_records: int = 4096):
        chunk_bytes = io_buffer_records * self.RECORD_SIZE
        with open(filename, 'rb') as f:
            while True:
                block = f.read(chunk_bytes)
                if not block:
                    break

                for i in range(0, len(block), self.RECORD_SIZE):
                    a,b,value = struct.unpack(self.RECORD_FMT, block[i:i+self.RECORD_SIZE])

                    yield (a,b), value

    def __iter__(self):
        pass

class GloVe():
    pass

if __name__ == '__main__':
    com = CoOccurrenceMatrix()
    a = com.calculate([[1,2,3,4,5], [5,4,3,2,1]], sliding_window=3, weighted=True, max_bytes_in_ram=192, check_ram_every_n_tokens=-1)
    print(a)
