import random

import numpy as np
from pyparsing import srange
from collections import defaultdict
from pympler import asizeof
import os
import struct
from pathlib import Path
import heapq
import tqdm as tqdm

class CoOccurrenceMatrix:
    BYTES_PER_PAIR = 192.0
    RECORD_FMT = "<iid"
    RECORD_SIZE = struct.calcsize(RECORD_FMT)

    def __init__(self, io_buffer_size: int = 4096):
        self.com_path: str = None
        self.shuffled_com_path: str = None
        self.io_buffer_size: int  = io_buffer_size

    # wighted - czy wyniki mają być ważone od odległości
    # max_bytes_in_ram - ile pamięci max będzie trzymana w ram
    #
    def calculate(self, tokenized_corpus: tuple[tuple[int]],
                  sliding_window: int = 3,
                  weighted: bool = True,
                  max_bytes_in_ram: int = 64 * 1024 * 1024,
                  check_ram_every_n_tokens: int = 20000,
                  shard_directory: str = "./shards_data"
                  ) -> None:
        shard_filenames: list[str] = []
        Path(shard_directory).mkdir(parents=True, exist_ok=True)

        buffer: dict[tuple[int,int], float] = defaultdict(float)
        tokens_since_check: int = 0

        max_pairs_estimate = max(1, int(max_bytes_in_ram / self.BYTES_PER_PAIR))

        progress_bar = tqdm.tqdm(range(len(tokenized_corpus)))
        for sentence in tokenized_corpus:
            n_length = len(sentence)

            for position in range(n_length):
                progress_bar.desc = f"Co-occurrence Matrix Calculating | Token Progress: {position + 1}/{n_length}"

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

            progress_bar.update(1)

        if buffer:
            shard_filenames.append(self._flush_buffer_to_shard(buffer, shard_directory=shard_directory, shard_idx=len(shard_filenames)))
            buffer.clear()

        progress_bar.close()

        self.com_path = os.path.join(shard_directory, 'merged_cooccurrence.bin')
        shards = [self._read_shard(f) for f in shard_filenames]

        progress_bar = tqdm.tqdm(range(len(shards)), desc="Co-occurrence Shards Processing | Processing...")
        with open(self.com_path, 'wb') as out:
            current_key, current_value = None, 0.0

            for key, value in heapq.merge(*shards, key=lambda kv: kv[0]):
                if key == current_key:
                    current_value += value

                else:
                    if current_key is not None:
                        out.write(struct.pack(self.RECORD_FMT, current_key[0], current_key[1], current_value))

                    current_key, current_value = key, value
                progress_bar.update(1)

            if current_key is not None:
                out.write(struct.pack(self.RECORD_FMT, current_key[0], current_key[1], current_value))

        progress_bar.close()
        progress_bar = tqdm.tqdm(desc="Co-occurrence Final Processing | Finishing...")

        for f in shard_filenames:
            progress_bar.desc = f"Co-occurrence Final Processing | Finishing..."
            os.remove(f)

            progress_bar.update(1)
        progress_bar.close()

    def shuffle(self, max_records_in_memory: int = 200_000,
                shard_directory: str = "./shards_data",
                seed: int = 42) -> None:
        random_number_generator = random.Random(seed)

        buffer: list = []
        shard_filenames: list[str] = []
        def flush(buffer):
            random_number_generator.shuffle(buffer)
            filename = os.path.join(shard_directory, f'shuffle_shard_{len(shard_filenames):05d}.bin')

            with open(filename, 'wb') as out:
                for record in buffer:
                    out.write(struct.pack(self.RECORD_FMT, *record))

            shard_filenames.append(filename)

        progress_bar = tqdm.tqdm(desc="Co-occurrence Shuffle Preparing | Processing...")
        with open(self.com_path, 'rb') as f:
            while True:
                record_chunk = f.read(self.RECORD_SIZE)
                if not record_chunk:
                    break

                buffer.append(struct.unpack(self.RECORD_FMT, record_chunk))
                if len(buffer) > max_records_in_memory:
                    flush(buffer)
                    buffer.clear()

                progress_bar.update(1)
            progress_bar.close()

        if buffer:
            flush(buffer)

        self.shuffled_com_path = os.path.join(shard_directory, "shuffled_cooccurrence.bin")

        readers = [open(f, 'rb') for f in shard_filenames]
        reader_indexes = list(range(len(readers)))
        progress_bar = tqdm.tqdm(desc="Co-occurrence Shuffling | Processing...")
        with open(self.shuffled_com_path, 'wb') as out:
            while reader_indexes:
                picked_index = random_number_generator.randrange(len(reader_indexes))
                index = reader_indexes[picked_index]

                reader_chunk = readers[index].read(self.RECORD_SIZE)
                if not reader_chunk:
                    readers[index].close()
                    reader_indexes.pop(picked_index)

                    continue

                out.write(reader_chunk)

                progress_bar.update(1)
            progress_bar.close()

        for f in shard_filenames:
            os.remove(f)

    def _flush_buffer_to_shard(self, buffer: dict, shard_directory: str, shard_idx: int) -> str:
        filename = os.path.join(shard_directory, f'com_shard_{shard_idx:05d}.bin')
        with open(filename, 'wb') as f:
            for (a, b), value in sorted(buffer.items()):
                f.write(struct.pack(self.RECORD_FMT, a, b, value))

        return filename

    def _read_shard(self, filename: str):
        chunk_bytes = self.io_buffer_size * self.RECORD_SIZE
        with open(filename, 'rb') as f:
            while True:
                block = f.read(chunk_bytes)
                if not block:
                    break

                for i in range(0, len(block), self.RECORD_SIZE):
                    a, b, value = struct.unpack(self.RECORD_FMT, block[i:i+self.RECORD_SIZE])

                    yield (a, b), value

    def __iter__(self):
        chunk_bytes = self.io_buffer_size * self.RECORD_SIZE
        with open(self.com_path, 'rb') as f:
            while True:
                block = f.read(chunk_bytes)
                if not block:
                    break

                for i in range(0, len(block), self.RECORD_SIZE):
                    a, b, value = struct.unpack(self.RECORD_FMT, block[i:i + self.RECORD_SIZE])

                    yield a, b, value

class GloVe:
    def __init__(self):
        self.co_occurrence_matrix: CoOccurrenceMatrix = CoOccurrenceMatrix()

    def calculate(self):
        pass

if __name__ == '__main__':
    com = CoOccurrenceMatrix()
    a = com.calculate([[1,2,3,4,5], [5,4,3,2,1]], sliding_window=3, weighted=True, max_bytes_in_ram=64 * 1000 * 1000, check_ram_every_n_tokens=-1)
    print(a)
    com.shuffle()
    print(com.shuffled_com_path)
    #
    # pos = 0
    # for b in com:
    #     print(pos, b)
    #     pos += 1
