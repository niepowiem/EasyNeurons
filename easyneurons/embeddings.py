import random

import numpy as np
from easyneurons.layer import RandomInitialization
from pyparsing import srange
from collections import defaultdict
from pympler import asizeof
import os
import struct
from pathlib import Path
import heapq
import tqdm as tqdm

RECORD_DTYPE = np.dtype([("i", "<i4"), ("j", "<i4"), ("value", "<f8")])
RECORD_SIZE = RECORD_DTYPE.itemsize

class CoOccurrenceMatrix:
    """
    Liczy i przechowuje macierz współwystępień poza pamięcią RAM
    """

    # Ile Byte RAM przypada na jeden rekord w buforze.
    # 16B Buffor,
    # 8B Permutacja z np.argsort
    # 16B Dwa gathery
    # Reszta na wynik agregacji
    # Szczyt RSS calculate to ~64B
    BYTES_PER_RECORD = 48

    # Po zagregowaniu buffora jeśli zostało więcej niż Tyle procet pojemnościm
    # kompresja się nie opłaca i rzucamy shard
    # Poniżej tego, zostawiamy dane w buforze i zbieramy dalej
    COMPACT_KEEP = 0.6

    def __init__(self, sliding_window: int = 5,
                 weighted: bool = True,
                 max_bytes_in_ram: int = 64 * 1024 * 1024,
                 shard_directory: str="./shards_data",
                 io_records: int = 1 << 16,
                 shuffle_records_per_shard: int = 1_000_000,
                 shuffle_interleave: int = 4096,
                 seed: int=None):

        self.com_path = None
        self.shuffled_com_path = None

        self.sliding_window = sliding_window
        self.weighted = weighted
        self.max_bytes_in_ram = max_bytes_in_ram
        self.shard_directory = shard_directory
        self.io_records = io_records
        self.shuffle_records_per_shard = shuffle_records_per_shard
        self.shuffle_interleave = shuffle_interleave
        self.seed = seed

    def calculate(self, tokenized_corpus: tuple[tuple[int]]):
        # Unieważniamy poprzednie wyniki
        self.com_path = None
        self.shuffled_com_path = None

        # Sprzątamy po ewentualnym, przerwanym poprzednim przebiegu
        # w przeciwnym wypadku stare shardy zostałyby wciągnięte do nowego com
        Path(self.shard_directory).mkdir(parents=True, exist_ok=True)
        for old_shards in glob.glob(os.path.join(self.shard_directory, 'com_shard_*.bin')):
            os.remove(old_shards)

        # Obliczmy, ile pojedynczych rekordów możemy zmieścić w pamięci
        # Zamiast listy Python, używani numpy zapewnia dyfajność i oszczędność pamięci
        # Licznik mówiący ile obecnie elementów znajduje się w buferze
        buffer_capacity = max(1024, self.max_bytes_in_ram // self.BYTES_PER_RECORD)
        buffer_key = np.empty(buffer_capacity, dtype=np.int64)
        buffer_value = np.empty(buffer_capacity, dtype=np.float64)
        buffered_count: int = 0

        # Tworzymy listę, w której będą przechowywane ścieżki do plików tymczasowych
        # Flaga której calem jest uruchomienie tylko dla pierwszego zdania oszczędzajać czas
        # na sprawdzanie poprawności danych dla każdego kolejnego zdania
        shard_filenames: list[str] = []
        first_sentence_validated = False

        for sentence in tokenized_corpus:

            # Konwertuje zdanie na tablicę numpy
            tokens = np.asarray(sentence)

            # Sprawdzamy, pierwsze zdanie, czy ma błedy w zapisie tokenów
            if not first_sentence_validated:
                self._validate(tokens=tokens)
                first_sentence_validated = True

            # Konwertuje elementy na 64bit liczby całkowite
            # Parametr copy=False optymalizje działanie, unikając niepotrzebnego kopiowania
            # danych w pamięci jeśli typ już jest odpowiedni
            tokens = tokens.astype(np.int64, copy=False)
            length = len(tokens)

            # Wylicza maksymalną odległośc o jaką możemy się przesunąć
            # Jeżeli zdanie składa się tylko z 1 tokena to pomijamy, ponieważ
            # nie da się na nim zbudować kontekstu
            max_distance = min(self.sliding_window, length - 1)
            if max_distance < 1:
                continue

            # Obliczamy ile par wygeneruje to zdanie
            # Potrzebujemy to, żeby sprawdzić miejsce PRZED zapisem,
            # a nie po, ponieważ mamy limit RAM
            # 2 *, ponieważ są dwie strony lewa i prawa
            sentence_pair_count = 2 * sum(length - distance for distance in range(1, max_distance + 1))

            # Jeżeli przekroczymy buffer scalamy powatarzające się elementy, aby zwolnić miejsce
            if buffered_count + sentence_pair_count > buffer_capacity:
                buffered_count = self._compact(buffer_key, buffer_value, buffered_count)

                # Jeżeli łączenie nie zwoliło za dużo mniejsca
                if buffered_count + sentence_pair_count > buffer_capacity * self.COMPACT_KEEP:

                    # Zrzucamy cały bufor na dysk
                    shard_filenames.append(self._write_shard(key=buffered_key[:buffered_count],
                                                             value=buffered_value[:buffered_count],
                                                             index=len(shard_filenames)))
                    buffered_count = 0

            for distance in range(1, max_distance + 1):

                # Wyliczamy liczbę par, którę powstaną z teskstu
                # Wyliczaby wagę na podstawie dystansu
                count = length - distance
                weight = 1.0 / distance if self.weighted else 1.0

                # Tworzymy lewą część tablicy i prawą
                left, right = tokens[:-distance], tokens[distance:]

                # Kodujemy parę tokenów jako jeden klucz
                # Token left trafia w górne 32 bity, a right w dowlne 32 bity (|)
                # Nie trzeba przez to trzymać pary jako krotki
                forward = (left << np.int64(32)) | right
                backward = (right << np.int64(32)) | left

                # Zapisujemy do buferów
                buffer_key[buffered_count: buffered_count + count] = forward
                buffer_value[buffered_count: buffered_count + count] = weight
                buffered_count += count

                buffer_key[buffered_count: buffered_count + count] = backward
                buffer_value[buffered_count: buffered_count + count] = weight
                buffered_count += count

            if buffered_count:
                # Zmniejszamy i scalamy pozostałe wartości w bufferze
                buffered_count = self._compact(buffer_key, buffer_value, buffered_count)

                # Zrzucamy cały bufor na dysk
                shard_filenames.append(self._write_shard(key=buffered_key[:buffered_count],
                                                         value=buffered_value[:buffered_count],
                                                         index=len(shard_filenames)))

            # Usuwamy duże tablize numpy z pamięci, które mogą zajmować sporo RAMu
            del buffer_key, buffer_value

            self.com_path = os.path.join(self.shard_directory, 'merged_cooccurrence.bin')
            self._merge_shards(shard_filenames, self.com_path)

            # Sprzątamy - usuwamy shardy
            for filename in shard_filenames:
                os.remove(filename)

            return self

    def shuffle(self):
        if self.com_path is None:
            raise RuntimeError("Najpierw wywolaj calculate()")

        # Inializujemy randomowe generowanie liczb z seed
        # Wyliczamy ile bajtów musimy wyczytać aby otrzymać daną ilość recordów
        # Inicjalizujemy listę nazw shardów
        rng = np.random.default_rng(self.seed)
        block_bytes = self.shuffle_records_per_shard * RECORD_SIZE
        shard_filenames = []

        # Otwieramy cooccurrence matrix obliczoną w calculate()
        with open(self.com_path, 'rb') as com:
            while True:

                # Czytamy blok po shuffle_records_per_shard rekorów
                # Jeżeli nic nie wyczytaliśmy, znaczy to, że to koniec pliku
                block = com.read(block_bytes)
                if not block:
                    break

                # Reinterpretujemy każdy blok jako tablica struct numpy
                # losowo permutujemy kolejność rekordów w obrębie tego jednego bloku (lokalne tasowanie)
                record = np.frombuffer(block, dtype=RECORD_DTYPE).copy()
                rng.shuffle(record)

                # Zapisujemy potasowany blok do nowego pliku tymczasowego
                filename = os.path.join(self.shard_directory, f"shuffle_shard_{len(shard_filenames):05d}.bin")
                with open(filename, 'wb') as f:
                    f.write(record.tobytes())

                # Dodajemy nazwę pliku do listy
                shard_filenames.append(filename)

        # Zapisujemy nazwę potasowanego com
        # Lista otwartych uchwytów do wszystkich potasowanych shardów
        # Lista indeksów shardów, które jeszcze mają dane do odczytania
        # Rozmiar kawałka danych czytnego na raz z pojedynczego shardu - jak mocno wymieszane są
        self.shuffled_com_path = os.path.join(self.shard_directory, "shuffled_cooccurrence.bin")
        shard_readers = [open(filename, 'rb') for filename in shard_filenames]
        shard_data_yet_to_read = list(range(len(shard_readers)))
        read_piece_bytes = self.shuffle_interleave * RECORD_SIZE

        with open(self.shuffled_com_path, 'wb') as f:

            # Dopóki nie wyczytamy wszystkich danych ze wszystkich shardów
            while shards_data_yet_to_read:

                # Losowo wybiera jeden z aktywnych shardów
                # mapuje losową pozycję na faktyczy numer shardu
                position = int(rng.integers(len(shards_data_yet_to_read)))
                index = shard_data_yet_to_read[position]

                # Czytamy kawałek piece_bytes bajtów
                # Jeżeli nic nie odczytaliśmy, to znaczy, że już nie ma żadnych danych do odczytu
                # Zamykamy reader i usuwamy idx z nie-pustych shardów
                # W przeciwnym wypadku zapisujemy blok w nowym, potasowanym pliku
                block = shard_readers[index].read(read_piece_bytes)
                if not block:
                    shard_readers[index].close()
                    shards_data_yet_to_read.pop(position)

                    continue

                f.write(block)

        # Czyścimy - usuwamy shardy
        for filename in shard_filenames:
            os.remove(filename)

        return self

    def _write_shard(self, key, value, index) -> str:
        i, j = self.__unpack_key(key=key)

        # Rezerwujemy miejsce w pamięci, a następnie zapisujemy wartości do struct
        record = np.empty(len(key), dtype=RECORD_DTYPE)
        record['i'] = i
        record['j'] = j
        record['value'] = value

        # Generujemy ścieżkę pliku i zrzucamy do pamięci dysku
        filename = os.path.join(self.shard_directory, f'com_shard_{index:05d}.bin')
        with open(filename, 'wb') as f:
            f.write(record.tobytes())

        return filename

    def _iterate_shard_pairs(self, filename, records_per_block):

        # Wyliczamy ile bajtów musimy odczytać, aby otrzymać dokładnie tyle rekordów
        block_bytes = records_per_block * self.RECORD_SIZE
        with open(filename, 'rb') as f:
            while True:
                block = f.read(block_bytes)

                # Jeżeli jest pust, oznacza to koniec pliku
                if not block:
                    break

                # Nie kopiujemy danych, twórzymy widok na surowych bajtach
                # interpretując je zgodnie ze strukturą RECORD_DTYPE
                record = np.frombuffer(block, dtype=RECORD_DTYPE)

                # Pakujemy klucze, onieważ w heapq wygoniej jest mieć jeden klucz
                # zamiast porównywać krotki
                yield from zip(self.__pack_key(record['i'],
                                               record['j']).tolist(),
                                               record['value'].tolist())

    def _merge_shards(self, shard_filenames, out_path) -> int:
        """
        Scalamy posortowane shardy w jeden pluk, sumując rekordy o tym samym kluczu
        """

        # Używamy zwykłych list python, ponieważ są szybsze niż numpy
        # Liczba zapisanych rekordów, zwracamy na końcu
        pending_buffer_key, pending_buffer_value = [], []
        writted_records = 0

        with open(out_path, "wb") as f:
            def dump():
                nonlocal writted_records
                if not pending_buffer_key:
                    return

                # Konwertujemy listę kluczy na numpy int64
                # Odwraca kodowanie poprzedniego fragmentu
                # (Rozdziela indeksty dwóch tokenów - górne i dolne 32 bity)
                key = np.asarray(pending_buffer_key, dtype=np.int64)
                i, j = self.__unpack_key(key=key)

                # Tworzymy struct numpy w formacie RECORD_DTYPE
                # czyli finalny format rekordu w pliku binarnym
                record = np.empty(len(key), dtype=RECORD_DTYPE)
                record['i'] = i
                record['j'] = j
                record['value'] = pending_buffer_value

                # Zapisujemy surowe bajty do pliku
                f.write(record.tobytes())

                # Aktualizujemy licznik i czyścimy bufory, żeby zacząć zbierać kolejny blok
                writted_records += 1
                pending_buffer_key.clear()
                pending_buffer_value.clear()

            # Wyliczamy rozmiar bloku, który będziemy odczytywać
            records_per_block = max(512, min(self.io_records, self.io_records // max(1, len(shard_filenames))))

            # Iterator czytający dane blokami records_per_block
            # żeby nie ładować całego shardo do pamięci naraz
            current_key, current_value = None, 0.0
            streams = (self._iterate_shard_pairs(f, records_per_block) for f in shard_filenames)

            # healpq.merge scala wszystkie posortowane strumienie w jeden posortowany strumień par (key, value)
            for key, value in heapq.merge(*streams):

                # Łączymy duplikaty między shardami
                # (Te same pary tokenów mogły trafić do różnych shardów)
                if key == current_key:
                    current_value += value

                else:

                    # Jeśli klucz się zmienił, oznacza to, że poprzednia grupa jest już kompletna
                    # Zapisujemy ją do bufora i zaczynamy nową grupę z nowym kluczem
                    if current_key is not None:
                        pending_buffer_key.append(current_key)
                        pending_buffer_value.append(current_value)

                        # Jeżeli bufor urośnie do self.io_records, zapisujemy go na dysk i zwalniamy pamięć
                        if len(pending_buffer_key) >= self.io_records:
                            dump()

                    current_key, current_value = key, value

            # Domykamy ostanią grupę i wykonujemy finajny dump aby zrzucić reszte bufora
            if current_key is not None:
                pending_buffer_key.append(current_key)
                pending_buffer_value.append(current_value)

            dump()

        # Zwracamy liczbę zapisanych rekordów w finalnym pliku
        return writted_records

    def read_batches(self, batch_size: int = 100_000):
        block_bytes = batch_size * RECORD_SIZE
        with open(self.path, "rb") as f:
            while True:

                # Czytamy blok o wskazaną ilość pozycji
                # Jeżeli nic nie wyczytaliśmy to znaczy,
                # że nic więcej w tym pliku nie ma
                block = f.read(block_bytes)
                if not block:
                    break

                record = np.frombuffer(block, dtype=RECORD_DTYPE)
                yield record["i"], record["j"], record["value"]

    @staticmethod
    def _validate(tokens: np.ndarray) -> None:
        # Sprawdzamy, poprawność tokenów, inaczej to bład wyskoczy dopiero przy zapisywaniu

        if tokens.dtype.kind not in 'iu':
            raise TypeError(f"Zdania musza zawierac calkowite id tokenow, otrzymano "f"dtype={tokens.dtype}")

        if tokens.size and (tokens.min() < 0 or tokens.max() > _INT32_MAX):
            raise ValueError(f"Id tokenow musza miescic sie w 0..{_INT32_MAX}, "
                             f"a zakres w pierwszym zdaniu to {tokens.min()}... {tokens.max()}")

    @staticmethod
    def _compact(buffer_key, buffer_value, buffered_count) -> int:
        if buffered_count == 0:
            return 0

        # Buffor może być większy niż liczba zgromadzonych w nim danych
        # Obejmyjemy tylko ten fragment pamięci, w którym są faktyczne dane
        key = buffer_key[:buffered_count]
        value = buffer_value[:buffered_count]

        # Aby pogrupować identyczne kludze, trzeba je posortować, wttedy znajdą się obok siebie w tablicy
        # Funkcja argsort zwraca indeksy posortowanej tablicy, dzięki czemu ułożone zostają i kludze i odpowiednie wartości
        order = np.argsort(key, kind='stable')
        key_sorted = key[order]
        value_sorted = value[order]

        # Znajdujemy granice powtarzających się kluczy
        # r_ łączy wiersz pod wiers [[],[]]; [[]] w [[],[],[]]
        # flatnonzero zamienia tablczę na konkretne indeksy miejsc
        boundries = np.flatnonzero(np.r_[True, key_sorted[1:] != key_sorted[:-1]])
        unique = len(boundries)

        # Bierzemy tylko te klucze, które zaczynają nową grupę
        # Sumujemy przeciał za pomocą wyznaczonych, pososrtowanych wartości
        buffer_key[:unique] = key_sorted[boundries]
        buffer_value[:unique] = np.add.reduceat(value_sorted, boundries)

        return unique

    @staticmethod
    def __pack_key(i, j):
        """(i, j) -> pojedynczy int64. Musi byc odwracalne i monotoniczne po (i, j)."""
        return (i.astype(np.int64) << np.int64(32)) | (j.astype(np.int64) & _LOW32)

    @staticmethod
    def __unpack_key(key):
        """int64 -> (i, j) jako int32."""
        return (key >> np.int64(32)).astype(np.int32), (key & _LOW32).astype(np.int32)

    @property
    def path(self) -> str:
        return self.shuffled_com_path if self.shuffled_com_path else self.com_path

    def __len__(self):
        return os.path.getsize(self.path) // RECORD_SIZE

    def __iter__(self):
        for i, j, value in self.read_batches(self.io_records):
            yield from zip(i.tolist(),
                           j.tolist(),
                           x.tolist())

class GloVe:
    def __init__(self, co_occurrence_matrix,
                 vocabulary_size: int,
                 d_model: int = 50,
                 occurrence_weight_max: int = 100,
                 alpha: float = 0.75,
                 max_batch_bytes:int = 256 * 1024 * 1024,
                 seed: int = None):
        self.co_occurrence_matrix = co_occurrence_matrix
        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.occurrence_weight_max = occurrence_weight_max
        self.alpha = alpha
        self.max_batch_bytes = max_batch_bytes
        self.seed = seed

if __name__ == '__main__':
    pass
