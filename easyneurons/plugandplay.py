from unittest import result

import marisa_trie
import numpy as np
from abc import ABC, abstractmethod
import unicodedata
import regex as re
import string

from pip._internal.cli import progress_bars
from tqdm import tqdm
import heapq
from collections import defaultdict, Counter
import itertools as it
from easyneurons.general import Pipeline


class MappedString():
    """
    This class is where normalized text is stored, aling with its alignments with the original text
    Simply use `variable_name`(start index, end index) to learn the original placement of char in the text
    """

    def __init__(self, string: str):
        self.string = string
        self.alignment = list(range(len(string)))
        self.original = string

    def __call__(self, start: int, end: int) -> tuple[int, int] | None:
        if end - 1 >= len(self.alignment) or end - 1 < 0:
            return None

        o_start = self.alignment[start]
        o_end = self.alignment[end - 1] + 1 if end > start else self.alignment[start]

        return (o_start, o_end)

    def __iter__(self):
        yield self

    def __len__(self):
        return len(self.string)

    def __repr__(self):
        return self.string


class BatchMappedString():
    """
    This class allows for multiple MappedString.s to enter nomalization instead of manually adding the MappedString class using loops
    """

    def __init__(self, strings: list[str] | str) -> None:
        if isinstance(strings, str):
            strings = (strings,)

        self.mapped_strings = [MappedString(string) for string in strings]

    def __iter__(self):
        return iter(self.mapped_strings)

    def __len__(self):
        return len(self.mapped_strings)

    def __repr__(self):
        return str([mapped_string.string for mapped_string in self.mapped_strings])


class Normalization(ABC):
    @abstractmethod
    def normalize(self, BatchMappedString) -> None:
        """
        Normalizuje obiekt w miejscu, nie tworząc kopii
        :param BatchMappedString:
        :return:
        """


class LowerCaseNormalization(Normalization):
    """
    Zamienia wszystkie upper-casy na lowercase text
    """

    @staticmethod
    def normalize(batch_mapped_string: BatchMappedString) -> None:
        for mapped_string in batch_mapped_string.mapped_strings:
            mapped_string.string = mapped_string.string.lower()


class WhitespaceNormalization(Normalization):
    """
    Usuwa nadmierne spacje np. "   " (3xSpacja) -> ' ' (1xSpacja)
    """

    @staticmethod
    def normalize(batch_mapped_string: BatchMappedString) -> None:
        for mapped_string in batch_mapped_string:
            if len(mapped_string.string) == 0:
                continue

            start = 0
            end = len(mapped_string.string)
            while start < end and mapped_string.string[start].isspace():
                start += 1

            while end > start and mapped_string.string[end - 1].isspace():
                end -= 1

            chars = []
            alignment = []
            previous_space = False
            for i in range(start, end):
                char = mapped_string.string[i]
                is_space = char.isspace()

                if is_space and previous_space:
                    continue

                chars.append(char)
                alignment.append(mapped_string.alignment[i])
                previous_space = is_space

            mapped_string.string = "".join(chars)
            mapped_string.alignment = alignment


class UnicodeNormalization(Normalization):
    """
    Normlaizuje znaki np.
    NFC właczy rozbite znaki (odwrotność NFD) ą [a + ˛ (2 znaki)] -> ą (1 znak)
    NFD rozbija znaki na wiele pod znaków (wygląda na tyle samo, ale zajmuje więcej bajtów) np.: ą -> ą [a + ˛ (2 znaki)]
    NFKC łączy rozbite znaki i upraszcza styl (robi to samo co NFC oraz...) 𝓗𝓮𝓵𝓵𝓸 𝓦𝓸𝓻𝓵𝓭 -> Hello World
    NFKD rozbija znaki na wiele pod znaków i upraszcza stył (robi to samo co NCKD oraz...) 𝓗𝓮𝓵𝓵𝓸 𝓦𝓸𝓻𝓵𝓭 -> Hello World
    """

    def __init__(self, mode: str = "NFC"):
        if mode not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError("Value of \'mode\' must be one of: NFC, NFD, NFK, NFKD")

        self.mode = mode

    def normalize(self, batch_mapped_string: BatchMappedString) -> None:
        for mapped_string in batch_mapped_string:
            n_text = unicodedata.normalize(self.mode, mapped_string.string)

            if len(n_text) == len(mapped_string.string):
                mapped_string.string = n_text
                continue

            n_text = ''
            n_alignment = []
            common = 0
            for i in range(len(mapped_string.string)):
                text_window = mapped_string.string[:i + 1]
                n_text_window = unicodedata.normalize(self.mode, text_window)

                while (common < min(len(n_text), len(n_text_window))
                       and n_text[common] == n_text_window[common]):
                    common += 1

                new_chars = n_text_window[common:]
                n_alignment.extend([i] * len(new_chars))
                n_text = n_text_window

            mapped_string.string = n_text
            mapped_string.alignment = n_alignment


class AccentsNormalization(Normalization):
    """
    Usuwa znaki diaktryczne / akcenty w literach takie jak: ą -> a, ó -> o, ś -> s... itp.
    """

    @staticmethod
    def normalize(batch_mapped_string: BatchMappedString) -> None:
        for mapped_string in batch_mapped_string:

            alignment: list[int] = []
            text: list[str] = []
            for i, char in enumerate(mapped_string.string):
                if unicodedata.category(char) == "Mn":
                    continue

                text.append(char)
                alignment.append(mapped_string.alignment[i])

            mapped_string.string = ''.join(text)
            mapped_string.alignment = alignment


class ReplaceNormalization(Normalization):
    """
    Zamienia kawałek tekstu na inny np. "Hugging Hug" (target = "Hug", replacement = "MUG") -> "MUGging MUG"
    """

    def __init__(self, target: str, replacement: str):
        if len(target) == 0:
            raise ValueError("\'target\' nie może być pustym stringiem")

        self.replacement = replacement
        self.target = target

    def normalize(self, batch_mapped_string: BatchMappedString) -> None:
        for mapped_string in batch_mapped_string:
            text = []
            alignment = []
            position = 0

            while True:
                i = mapped_string.string.find(self.target, position)

                if i == -1:
                    text.extend(mapped_string.string[position:])
                    alignment.extend(mapped_string.alignment[position:])
                    break

                text.extend(mapped_string.string[position:i])
                alignment.extend(mapped_string.alignment[position:i])

                last_origin = mapped_string.alignment[i + len(self.target) - 1]
                text.extend(self.replacement)
                alignment.extend([last_origin] * len(self.replacement))

                position = i + len(self.target)

            mapped_string.string = "".join(text)
            mapped_string.alignment = alignment


class NormalizationSequence(Normalization):
    """
    Pozwala połączyć wiele normalizatorów w sekwencji
    """

    def __init__(self, normalizers: list[Normalization]) -> None:
        self.normalizers = normalizers

    def normalize(self, batch_mapped_string: BatchMappedString) -> BatchMappedPreTokens:
        for normalizer in self.normalizers:
            normalizer.normalize(batch_mapped_string)

        return BatchMappedPreTokens(batch_mapped_string)


class MappedPreTokens():
    def __init__(self, mapped_string: MappedString) -> None:
        # Pierwszy wariant to przed SubTokenizer, a drugi to ten po
        self.pre_tokens: list[str | tuple(str | int)] = [mapped_string.string]
        self.alignment: list[list[int] | tuple[tuple[int]]] = [mapped_string.alignment]
        self.original: str = mapped_string.original

    def __iter__(self):
        yield self

    def __len__(self):
        return len(self.pre_tokens)

    def __repr__(self):
        return str(self.pre_tokens)


class BatchMappedPreTokens():
    def __init__(self, batch_mapped_string: BatchMappedString) -> None:
        self.mapped_pre_tokens = [MappedPreTokens(mapped_string) for mapped_string in batch_mapped_string]

    def __iter__(self):
        return iter(self.mapped_pre_tokens)

    def __len__(self):
        return len(self.mapped_pre_tokens)

    def __repr__(self):
        return str([single_mapped_pre_tokens.pre_tokens for single_mapped_pre_tokens in self.mapped_pre_tokens])


class PreTokenizer(ABC):

    @abstractmethod
    def pre_tokenize(self, batch_mapped_pre_tokens: BatchMappedPreTokens) -> None:
        pass


class RegexPreTokenizer(PreTokenizer):
    GPT2 = "|".join([
        r"'(?:[sdmt]|ll|ve|re)",  # T-1, English contractions
        r" ?\p{L}+",  # T-2, words
        r" ?\p{N}+",  # T-3, digits
        r" ?[^\s\p{L}\p{N}]+",  # T-4, not letters, digits, or whitespace
        r"\s+(?!\S)",  # T-5, all-but-last whitespace
        r"\s+"  # T-6, whitespace
    ])

    GPT4 = "|".join([
        r"'(?i:[sdmt]|ll|ve|re)",  # F-1, English contractions
        r"[^\r\n\p{L}\p{N}]?+\p{L}+",  # F-2, words, w/ opt non-alphanumeric
        r"\p{N}{1,3}",  # F-3, digits
        r" ?[^\s\p{L}\p{N}]++[\r\n]*",  # F-4, not letters, digits, or whitespace
        r"\s*[\r\n]",  # F-5, whitespace with line-ending
        r"\s+(?!\S)",  # F-6, all-but-last whitespace
        r"\s+"  # F-7, all whitespace
    ])

    GPT4o = "|".join([
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*\
        [\p{Ll}\p{Lm}\p{Lo}\p{M}]+\
        (?i:'s|'t|'re|'ve|'m|'ll|'d)?",  # O-1 word with some lowercase
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+\
        [\p{Ll}\p{Lm}\p{Lo}\p{M}]*\
        (?i:'s|'t|'re|'ve|'m|'ll|'d)?",  # O-2 word with some uppercase
        r"\p{N}{1,3}""",  # O-3, digits
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",  # O-4, not letters, digits, or whitespace
        r"\s*[\r\n]+",  # O-5, whitespace with line-ending
        r"\s+(?!\S)",  # O-6, all-but-last whitespace
        r"\s+",  # O-7, all whitespace
    ])

    # https://arxiv.org/pdf/2504.00178
    BOUNDLESS = "|".join([
        r" ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+",  # B-1, contraction
        r"_(?:\p{Ll}\p{M}*)+",  # B-2, snake_case
        r" ?(?:\p{Lu}\p{M}*)+(?=(?:\p{Lu}\p{M}*)(?:\p{Ll}\p{M}*))",  # B-3, words
        r" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+",  # B-4, words
        r" ?(?:\p{Lu}\p{M}*)+",  # B-5, words
        r" ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+",  # B-6, words
        r"(?:\p{N}\p{M}*){1,3}(?=(?:(?:\p{N}\p{M}*){3})*(?:(?:\P{N}\p{M}*)|$))",  # B-7
        r" ?(?:[\p{P}\p{S}]\p{M}*)+",  # B-8, punct and symbols
        r"[^\S\r\n]*[\n\r]+|[^\S\r\n]+",  # B-9, whitespace
        r"(?:[\p{Z}\p{C}]\p{M}*)+",  # B-10, sep or control
        r"\p{M}+"  # B-11, leftover marks
    ])

    """
    Pozwala na dzielenie tekstu na tokeny wedle regexu (standardowo regex GPT2)
    keep_matches = True dzieli znalezione na chunki tekstu (pre-tokenizer)
    keep_matches = False usuwa znalezione char'y
    """

    def __init__(self, regex: str = GPT2, keep_matches: bool = True) -> None:
        self.regex = regex
        self.compiled = re.compile(regex)
        self.keep_matches = keep_matches

    def pre_tokenize(self, batch_mapped_pre_tokens: BatchMappedPreTokens) -> None:
        for mapped_pre_tokens in batch_mapped_pre_tokens:
            new_tokens = []
            new_alignment = []

            for token_text, token_alignment in zip(mapped_pre_tokens.pre_tokens, mapped_pre_tokens.alignment):
                position = 0

                for match in self.compiled.finditer(token_text):
                    start, end = match.span()

                    if start > position:
                        new_tokens.append(token_text[position:start])
                        new_alignment.append(token_alignment[position:start])

                    if end > start and self.keep_matches:
                        new_tokens.append(token_text[start:end])
                        new_alignment.append(token_alignment[start:end])

                    position = end

                if position < len(token_text):
                    new_tokens.append(token_text[position:])
                    new_alignment.append(token_alignment[position:])

            mapped_pre_tokens.pre_tokens = new_tokens
            mapped_pre_tokens.alignment = new_alignment


class WhitespacePreTokenizer(PreTokenizer):
    """
    Dzieli tekst na pre tokeny po wykrytej spacji np. Lorem ipsum dolor sit amed -> ['Lorem', 'ipsum', 'dolor', 'sit', 'amed.']
    """

    @staticmethod
    def pre_tokenize(batch_mapped_pre_tokens: BatchMappedPreTokens) -> None:
        RegexPreTokenizer(r"\s+", keep_matches=False).pre_tokenize(batch_mapped_pre_tokens)


class PunctuationPreTokenizer(PreTokenizer):
    """
    Dzieli tekst co wykryty znak punkcyjny np. Hugging Hug Hugged, Hugg Huggingface Face. -> ['Hugging Hug Hugged', ',', ' Hugg Huggingface Face', '.']
    """

    @staticmethod
    def pre_tokenize(batch_mapped_pre_tokens: BatchMappedPreTokens) -> None:
        RegexPreTokenizer(f"[{re.escape(string.punctuation)}]").pre_tokenize(batch_mapped_pre_tokens)


class ByteLevelPreTokenizer(PreTokenizer):
    """
    Zamienia znaki na UTF-8, i przekształca te bajty UTF-8 z powrotem na czytelne znaki tekstowe (znaki Unicode)
    np. 怨ン縁 ->æĢ¨ãĥ³ç¸ģ
    np. ゔみバ ->ãĤĶãģ¿ãĥĲ
    np. 𝓗𝓮𝓵𝓵𝓸 𝓦𝓸𝓻𝓵𝓭 -> ðĿĵĹðĿĵ®ðĿĵµðĿĵµðĿĵ¸ĠðĿĵ¦ðĿĵ¸ðĿĵ»ðĿĵµðĿĵŃ
    np. Lorem ipsum dolor sit amed. -> LoremĠipsumĠdolorĠsitĠamed

    Dzieki temu przy odpowiednio wytrenowanym tokenizerze nie ma potrzeby na token <UNK>
    """

    def __init__(self):
        printable = (list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)))
        self.byte_to_char = {byte: chr(byte) for byte in printable}

        next_code = 256
        for byte in range(256):
            if byte not in self.byte_to_char:
                self.byte_to_char[byte] = chr(next_code)
                next_code += 1

    def pre_tokenize(self, batch_mapped_pre_tokens: BatchMappedPreTokens) -> None:
        for mapped_pre_tokens in batch_mapped_pre_tokens:
            for t_idx in range(len(mapped_pre_tokens.pre_tokens)):
                token = mapped_pre_tokens.pre_tokens[t_idx]

                char_alignment = []
                for i, char in enumerate(token):
                    char_alignment.extend([i] * len(char.encode("utf-8")))

                mapped_pre_tokens.pre_tokens[t_idx] = ''.join(self.byte_to_char[byte] for byte in token.encode("utf-8"))
                mapped_pre_tokens.alignment[t_idx] = [mapped_pre_tokens.alignment[t_idx][i] for i in char_alignment]


class PreTokenizerSequence(PreTokenizer):
    """
    Pozwala połączyć wiele PreTokenizerów w sekwencji

    w przeciwieństwie do NormalizationSequence tutaj nie zwracamy kolejnej klasy ponieważ następne w kolejce jest SubWordTokenizer taki jak: BPE, WP, Unigram
    """

    def __init__(self, pre_tokenizers: List[PreTokenizer]) -> None:
        self.pre_tokenizers = pre_tokenizers

    def pre_tokenize(self, batch_mapped_pre_tokens: BatchMappedPreTokens) -> None:
        for pre_tokenizer in self.pre_tokenizers:
            pre_tokenizer.pre_tokenize(batch_mapped_pre_tokens)


class Vocabulary():
    """
    Zawiera zbiór wszystkich tokenów
    id to token i token to id
    """

    # WAŻNE!!!
    # Unknown token ZAWSZE musi być w indeksie 0
    def __init__(self, special_tokens: dict[str, str] = None) -> None:
        if special_tokens is None:
            self.special_tokens: dict[str, str] = {
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "sep_token": "<SEP>",
                "bos_token": "<BOS>",
                "eos_token": "<EOS>",

                # Optional
                "mask_token": "[MASK]",
                "cls_token": "[CLS]",
            }

        self.tokens = list(special_tokens.values())
        self.token_to_id: dict[str, int] = {k: v for v, k in enumerate(self.tokens)}

    # Dodaje token i zwraca jego index, jeżeli token już istnieje w vocabulary to nie dodaje ale i tak zwraca index
    def add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]

        id = len(self.tokens)

        self.token_to_id[token] = id
        self.tokens.append(token)

        return id

    # Wskazuje index tokenu (str)
    # lub
    # Mówi jakie id ma jaki token
    def get(self, token: str | int) -> int | str:
        if isinstance(token, str):
            if token in self.token_to_id:
                return self.token_to_id[token]
            else:
                return 0  # Zwracamy token [UNK]

        else:
            return self.tokens[token]

    def save(self, name: str = None, path: str = None) -> str:
        vocabulary_data = {
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
        }

        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"vocabulary_{timestamp}"

        if not path:
            path = "vocabularies"

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{name}.pkl")

        with open(filepath, 'wb') as f:
            pickle.dump(vocabulary_data, f)

        return filepath

    @classmethod
    def load(cls, filename: str = None, path: str = 'vocabularies') -> Vocabulary:
        if not filename:
            raise ValueError("Vocabulary filename not specified!")

        filepath = os.path.join(path, f"{filename}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'rb') as f:
            vocabulary_data = pickle.load(f)

        vocabulary = cls.__new__(cls)

        vocabulary.special_tokens = vocabulary_data["special_tokens"]
        vocabulary.token_to_id = vocabulary_data["token_to_id"]
        vocabulary.tokens = list(vocabulary.token_to_id.keys())

        return vocabulary

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id


class SubwordTokenizer(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def tokenize(self):
        pass

    @abstractmethod
    def encode(self, batch_mapped_pre_tokens: BatchMappedPreTokens) -> BatchProcessTokens:
        pass

    def decode(self, batch_tokens: BatchMappedPreTokens | BatchProcessTokens) -> tuple[tuple[str] | str]:
        batch_decoded = []

        if isinstance(batch_tokens, BatchMappedPreTokens):
            for mapped_pre_tokens in batch_tokens:
                decoded = []

                for pre_token_chunk in mapped_pre_tokens.pre_tokens:
                    chunk = []

                    for token in pre_token_chunk:
                        chunk.append(self.vocabulary.get(token))

                    decoded.append(tuple(chunk))
                batch_decoded.append(tuple(decoded))

        elif isinstance(batch_tokens, BatchProcessTokens):
            for processed_tokens in batch_tokens:
                decoded = []

                for token in processed_tokens.tokens:
                    decoded.append(self.vocabulary.get(token))

                batch_decoded.append(tuple(decoded))

        return tuple(batch_decoded)

    @abstractmethod
    def save(self):
        pass

    @classmethod
    @abstractmethod
    def load(self) -> SubwordTokenizer:
        pass

    @staticmethod
    def _merge_pairs_in_tokens(token: list[str], best_pair: tuple[str, str], merged: str) -> None:
        out: list[str] = []

        i = 0
        n = len(token)
        while i < n:
            if i < n - 1 and token[i] == best_pair[0] and token[i + 1] == best_pair[1]:
                out.append(merged)
                i += 2

            else:
                out.append(token[i])
                i += 1

        return out


# Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016
# https://arxiv.org/abs/1508.07909
# Included Dropout
class BPE(SubwordTokenizer):
    end_of_token_suffix = '</s>'

    def __init__(self, vocabulary: Vocabulary = None) -> None:
        self.vocabulary = vocabulary if vocabulary is not None else Vocabulary()

        # Zaiera dodane pary
        self.merges: list[tuple[str, str]] = []

        # Ważność / priorytet przy tokenizacji
        self.ranks: dict[tuple[str, str], int] = {}

        # Przy tokenizacji, jeżeli program napotka słowo, które już wcześniej widziało,
        # wyciągnie go z cache od razu oszczędzając czas
        self.cache: dict[str, tuple[tuple[str]]] = {}
        self.cache_limit: int = 1 << 20

    def train(self, batch_mapped_pre_tokens: BatchMappedPreTokens, merges: int | float = 0.35) -> None:
        every_token = tuple(
            pre_tokens for mapped_pre_tokens in batch_mapped_pre_tokens for pre_tokens in mapped_pre_tokens.pre_tokens)

        # Bierzemy wszystkie listy tokenów, ze wszystkich batch_mapped_pre_tokens i łączymy je w jedną listę i liczymy ilość słów
        initial_counts = Counter(every_token)

        # Automatycznie wyliczamy ilość mergów (nie dokładnie) jeżeli jest float
        if merges < 1:
            merges = len(initial_counts) * merges

        # tokeny rozbite na osobne char
        # Zawiera ilośc występowań każdego z tokenu
        tokens: list[list[str]] = []
        frequency: list[int] = []

        # Zapisujemy ile razy pojawia się dany char
        # zapisujemy tutaj każdy poszczegulny char, aby dodać go do vocabulary
        char_counts: dict[str, int] = defaultdict(int)
        alphabet: set[str] = set()

        # Dodajemy [a,b,c,d,e,f,g] (token) + [suffix] = [a,b,c,d,e,f,g,suffix]
        suffix = [self.end_of_token_suffix]
        for token, count in initial_counts.items():
            tokens.append(list(token) + suffix)
            frequency.append(count)

            # Dodajemy słowo, które nastepnie jest rozkładane na każdą poszczególną lierkę
            alphabet.update(token)

        # Dodajemy wszystkie znaki do vocabulary oraz suffix
        for char in sorted(alphabet):
            self.vocabulary.add(char)
        self.vocabulary.add(self.end_of_token_suffix)

        # Licznik par - ile razy dana para występuje
        # Ile razy para występuje we wskazanej patrze
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_occurances: dict[tuple[str, str], set[int]] = defaultdict(set)

        # token = ['ゔ', 'み', 'バ', '</s>']
        ######################
        ### Spróbuj przerzucić to do for token, count
        for idx, token in enumerate(tokens):
            word_frequency = frequency[idx]
            for pair in zip(token, token[1:]):
                # Wskazuje ile razy pojawia się para
                # Wskazuje w których tokenach pojawia się dana para (zmienna tokens)
                pair_counts[pair] += word_frequency
                pair_occurances[pair].add(idx)

        # heap sortuje od najmniejszego idx do największego (my chcemy od najwiekszego do najmniejszego dlatego - przy count)
        # tworząc coś podobnego do binary tree, oraz oddając nam najleszą lekograficznieparę
        pairs_heap: list[tuple[int, tuple[str, str]]] = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(pairs_heap)

        # Wykonujemy łącznie par i dodawnaie do vocabulary
        progress_bar = tqdm(range(int(merges)), desc="BPE(TBA)")
        for step in range(int(merges)):
            best_pair: tuple[str, str] = None

            # Omijamy jeżeli heap jest pusty - nie ma nic do łączenia
            while pairs_heap:
                # Otrzymujemy najlepszą parę
                count, pair = heapq.heappop(pairs_heap)

                # Sprawdzamy, czy para nadal jest aktualna
                if pair_counts.get(pair, 0) == -count:
                    best_pair = pair
                    progress_bar.desc = f"BPE(best_pair: {best_pair} x{-count})"
                    break

            # Jeżeli nie ma już czego mergować
            if best_pair is None:
                break

            # Zapisujemy pary jako para i merge w vocabulary
            self.merges.append(best_pair)
            merged_pair = best_pair[0] + best_pair[1]
            self.vocabulary.add(merged_pair)

            # Pary, które zostaną zmienione (dodane, usunięte lub zmiejszona ilośc występowań)
            change_affected_pairs: set[tuple[str, str]] = set()

            # Wciągamy indeksy tokenów, w których jest para
            tokens_with_pair_ids = pair_occurances.pop(best_pair, set())
            for idx in tokens_with_pair_ids:

                # Zmergowano parę w tokenie
                updated_token = self._merge_pairs_in_tokens(tokens[idx], best_pair, merged_pair)

                # Wyliczamy ilośc nowych i starych par
                old_pairs = Counter(zip(tokens[idx], tokens[idx][1:]))
                new_pairs = Counter(zip(updated_token, updated_token[1:]))

                for pair in old_pairs.keys() | new_pairs.keys():
                    # Obliczamy czy coś się zmieniło
                    # Jeżeli 0 - nic się nie zmieniło
                    # Jeżeli 1<= Dodano nową parę
                    # Jeżeli 0>= Usunięto parę (nowa zastapiła część starych par)
                    delta = (new_pairs[pair] - old_pairs[pair]) * frequency[idx]

                    # Jeżeli coś się zmieniło (nie jest 0)
                    if delta:
                        # Odejmujemy lub dodajemy ilośc wystąpień pary w zależności od zmiany
                        pair_counts[pair] += delta
                        change_affected_pairs.add(pair)

                    # Jeżeli para występuje w new_pairs, dodajemy jej wystąpowania
                    if new_pairs[pair]:
                        pair_occurances[pair].add(idx)

                    # Jeżeli para nie występuje w nowych parach, to usuwamy jej miejsce występowania (bo jej już tam nie ma)
                    else:
                        pair_occurance = pair_occurances[pair]
                        if pair_occurance is not None:
                            pair_occurance.discard(idx)

                # Aktualizujemy token, dając mu jego zmergowaną wersję
                tokens[idx] = updated_token

            # Sprawdzamy pary, które zostały zmienione (dodane, usunięte lub zmiejszona ilośc występowań)
            for pair in change_affected_pairs:

                # Sprawdzamy ile jest występowań pary
                count = pair_counts.get(pair, 0)

                # Jeżeli para już nie wystepuje, usuwamy ją z counts i occurances
                if count <= 0:
                    pair_counts.pop(pair, None)
                    pair_occurances.pop(pair, None)

                else:
                    heapq.heappush(pairs_heap, (-count, pair))

            progress_bar.update(1)
        progress_bar.close()

        # Budujemy ranks i czyścimy cache
        self._build_ranks()

    def _build_ranks(self):

        # Budujemy ranks i czyścimy cache
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.cache.clear()

    # Dropout = False przy inferencji
    # Dropout = True podczas treningu, typowo 0.05 - 0.1
    def tokenize(self, batch_mapped_pre_tokens: BatchMappedPreTokens, encode: bool = False, dropout: float = 0.0) -> None:
        suffix = [self.end_of_token_suffix]

        for mapped_pre_tokens in batch_mapped_pre_tokens:
            tokenized_tokens: list[tuple[str | int, ...]] = []
            tokenized_alignments: list[tuple[int, ...]] = []

            # single_token to pojedynczy token / słowo np. Huggingface
            for single_token, alignment in zip(mapped_pre_tokens.pre_tokens, mapped_pre_tokens.alignment):

                # Jeżeli token był już tokenizowany, po postu wyciągamy go z cache tysamym oszczędzając czas
                # przy dropout>0 segmentacja MA byc losowa, wiec cache pomijamy calkowicie.
                chars = None if dropout > 0.0 else self.cache.get(single_token, None)
                if chars is None:

                    # Nie udało się pozyskać cache, robimy od nowa na piechotę
                    # Tworzymy charactery z tokena ([a,b,c,d,e,f,suffix])
                    chars: list[str] = list(single_token) + suffix
                    while len(chars) > 1:
                        best_pair: tuple[str, str] = None
                        best_rank_found = len(self.ranks)
                        for pair in zip(chars, chars[1:]):

                            # Jeżel dropout większy to zachowujemy się jakty tej pary nie było w tabeli mergów
                            if dropout > 0.0 and random.random() < dropout:
                                continue

                            # Sprawdzamy czy para jest w mergach, jeżeli jej nie ma to pomijamy ten kod
                            # i przechodzimy do zmergowania jest w self.merge_pairs...
                            # Jeżeli jest aktualizujemy best pair i jeżeli nie znajdziemy lepszej, wtedy mergujemy
                            pair_rank = self.ranks.get(pair)
                            if pair_rank is not None and pair_rank < best_rank_found:
                                best_pair, best_rank_found = pair, pair_rank

                        # Jeżeli nie jesteśmy znaleźć kolejnej pary
                        if best_pair is None:
                            break

                        # Mergujemy pary
                        chars = tuple(self._merge_pairs_in_tokens(chars, best_pair, best_pair[0] + best_pair[1]))

                        # Dodajemy token do cache (Jeżeli nie dropoutujemy)
                        if dropout == 0.0 and len(self.cache) < self.cache_limit:
                            self.cache[single_token] = chars

                # Inicjalizuje zmienną z wynikami tokenizacji
                # Inicjalizuje zmienna z alignmentem tokenu
                token_result: list[str | int, ...] = []
                token_alignment: list[tuple[int, ...]] = []

                # Obliczamy alignment sub_token po sub_tokenie ('Hug', 'g', 'in', 'g</s>')
                position = 0
                for sub_token in chars:
                    # Jeżeli chcemy od razu enkodować (zamienić na vocab id)
                    if encode:
                        token_result.append(self.vocabulary.get(sub_token))

                    # Obliczamy odpowiendnio alignment
                    sub_token_span = len(sub_token)
                    if self.end_of_token_suffix in sub_token:
                        sub_token_span -= len(self.end_of_token_suffix) * sub_token.count(self.end_of_token_suffix)

                    token_alignment.append(tuple(alignment[position:position + sub_token_span]))
                    position += sub_token_span

                # Jeżeli nie wybrano enkodowania
                token_result = chars if not encode else tuple(token_result)

                # Doajemy do tokenów
                tokenized_tokens.append(token_result)
                tokenized_alignments.append(tuple(token_alignment))

            # Zapisujemy tokeny i aligment do obiektu
            mapped_pre_tokens.pre_tokens = tokenized_tokens
            mapped_pre_tokens.alignment = tokenized_alignments

    def encode(self, batch_mapped_pre_tokens: BatchMappedPreTokens, dropout: float = 0.0) -> BatchProcessTokens:
        self.tokenize(batch_mapped_pre_tokens, encode=True, dropout=dropout)
        return BatchProcessTokens(batch_mapped_pre_tokens)

    def save(self, name: str = None, path: str = None) -> str:
        vocabulary_data = {
            "special_tokens": self.vocabulary.special_tokens,
            "token_to_id": self.vocabulary.token_to_id,
        }

        subword_tokenizer_data = {
            "vocabulary_data": vocabulary_data,
            "merges": self.merges
        }

        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"bpe_{timestamp}"

        if not path:
            path = "subword_tokenizers"

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{name}.pkl")

        with open(filepath, 'wb') as f:
            pickle.dump(subword_tokenizer_data, f)

        return filepath

    @classmethod
    def load(cls, filename: str = None, path: str = 'subword_tokenizers') -> BPE:
        if not filename:
            raise ValueError("Subword Tokenizer filename not specified!")

        filepath = os.path.join(path, f"{filename}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'rb') as f:
            subword_tokenizer_data = pickle.load(f)

        vocabulary = Vocabulary.__new__(Vocabulary)
        vocabulary.special_tokens = subword_tokenizer_data["vocabulary_data"]["special_tokens"]
        vocabulary.token_to_id = subword_tokenizer_data["vocabulary_data"]["token_to_id"]
        vocabulary.tokens = list(vocabulary.token_to_id.keys())

        bpe = cls.__new__(cls)
        bpe.vocabulary: Vocabulary = vocabulary
        bpe.merges: list[tuple[str, str]] = subword_tokenizer_data["merges"]
        bpe.ranks: dict[tuple[str, str], int] = {}
        bpe._build_ranks()

        bpe.cache: dict[str, tuple[tuple[str]]] = { }
        bpe.cache_limit: int = 1 << 20

        return bpe


# Craig W. Schmidt, Varshini Reddy & Chris Tanner 2025
# https://arxiv.org/pdf/2504.00178
class BoundlessBPE(SubwordTokenizer):
    pass


# Alisa Liu, Jonathan Hayase, Valentin Hofmann, Sewoong Oh, Noah A. Smith︎, Yejin Choi 2025
# https://arxiv.org/pdf/2503.13423
class SuperBPE(SubwordTokenizer):
    pass


# https://arxiv.org/pdf/2106.12672
class WordPiece(SubwordTokenizer):
    # Zamiast dodawania ## do każdego podzielonego słowa, po prostu damy </b> na początku
    beggining_of_token_suffix = '</b>'

    def __init__(self, vocabulary: Vocabulary = None) -> None:
        self.vocabulary = vocabulary if vocabulary is not None else Vocabulary()

        # Tworzy trie do szybkiego greedy tokenization
        # Jeżeli token jest >100 to zastępujemy go UNK
        self.trie: marisa_trie.trie = None
        self.max_input_chars_per_word: int = 100

        # Przy tokenizacji, jeżeli program napotka słowo, które już wcześniej widziało,
        # wyciągnie go z cache od razu oszczędzając czas
        self.cache: dict[str, tuple[tuple[str]]] = {}
        self.cache_limit: int = 1 << 20

    def train(self, batch_mapped_pre_tokens: BatchMappedPreTokens, merges: int | float = 0.35) -> None:
        every_token = tuple(
            pre_tokens for mapped_pre_tokens in batch_mapped_pre_tokens for pre_tokens in mapped_pre_tokens.pre_tokens)

        # Bierzemy wszystkie listy tokenów, ze wszystkich batch_mapped_pre_tokens i łączymy je w jedną listę i liczymy ilość słów
        initial_counts = Counter(every_token)

        # Automatycznie wyliczamy ilość mergów (nie dokładnie) jeżeli jest float
        if merges < 1:
            merges = len(initial_counts) * merges

        # tokeny rozbite na osobne char
        # Zawiera ilośc występowań każdego z tokenu
        tokens: list[list[str]] = []
        frequency: list[int] = []

        # zapisujemy tutaj każdy poszczegulny char, aby dodać go do vocabulary
        alphabet: set[str] = set()

        # Dodajemy [a,b,c,d,e,f,g] (token) + [suffix] = [a,b,c,d,e,f,g,suffix]
        suffix = [self.beggining_of_token_suffix]
        for token, count in initial_counts.items():
            tokens.append(suffix + list(token))
            frequency.append(count)

            # Dodajemy słowo, które nastepnie jest rozkładane na każdą poszczególną lierkę
            alphabet.update(token)

        # Dodajemy wszystkie znaki do vocabulary oraz suffix
        for char in sorted(alphabet):
            self.vocabulary.add(char)
        self.vocabulary.add(self.beggining_of_token_suffix)

        # Licznik par - ile razy dana para występuje
        # Ile razy para występuje we wskazanej patrze
        # Ile razy dany char występuje
        # Wpisujemy sumbol i otrzymujemy jakie pary tworzy
        # Obliczone wyniki dla pary
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_occurances: dict[tuple[str, str], set[int]] = defaultdict(set)
        char_counts: dict[str, int] = defaultdict(int)
        symbol_to_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
        scores: dict[tuple[str, str], int] = defaultdict(int)

        # heap sortuje od najmniejszego idx do największego (my chcemy od najwiekszego do najmniejszego dlatego - przy count)
        # tworząc coś podobnego do binary tree, oraz oddając nam najleszą lekograficznieparę
        pairs_heap: list[tuple[int, tuple[str, str]]] = []

        # Heapujemy listę
        # heapq.heapify(pairs_heap)

        # Funkcja, która oblicza wynik dla pary i dodaje do heap
        def recalculate_score(pair: tuple[str, str]) -> None:
            score = pair_counts[pair] / (char_counts[pair[0]] * char_counts[pair[1]])
            scores[pair] = score
            heapq.heappush(pairs_heap, (-score, pair))

        # token = ['ゔ', 'み', 'バ', '</s>']
        for idx, token in enumerate(tokens):
            word_frequency = frequency[idx]

            # Liczymy ile takich char pojawia sie w tokenach
            for char in token:
                char_counts[char] += word_frequency

            for pair in zip(token, token[1:]):
                # Wskazuje ile razy pojawia się para
                # Wskazuje w których tokenach pojawia się dana para (zmienna tokens)
                pair_counts[pair] += word_frequency
                pair_occurances[pair].add(idx)

                # Dodajemy informacje jakie pary tworzy symbol
                symbol_to_pairs[pair[0]].add(pair)
                symbol_to_pairs[pair[1]].add(pair)

        # Obliczamy wyniki i budujemy heap
        for pair in pair_counts:
            recalculate_score(pair)

        progress_bar = tqdm(range(int(merges)), desc="WordPiece(TBA)")
        for step in range(int(merges)):
            best_pair: tuple[str, str] = None

            # Omijamy jeżeli heap jest pusty - nie ma nic do łączenia
            while pairs_heap:
                # Otrzymujemy najlepszą parę
                score, pair = heapq.heappop(pairs_heap)

                # Sprawdzamy, czy para nadal jest aktualna
                if pair in scores and scores[pair] == -score:
                    best_pair = pair
                    progress_bar.desc = f"BPE(best_pair: {best_pair} score={-score})"
                    break

            # Jeżeli nie ma już czego mergować
            if best_pair is None:
                break

            # Mergujemy parę i zapisujemy do vocabulary
            merged_pair = best_pair[0] + best_pair[1]
            self.vocabulary.add(merged_pair)

            # Pary, które zostaną zmienione (dodane, usunięte lub zmiejszona ilośc występowań)
            change_affected_pairs: set[tuple[str, str]] = set()

            # Wciągamy indeksy tokenów, w których jest para
            tokens_with_pair_ids = pair_occurances.pop(best_pair, set())
            for idx in tokens_with_pair_ids:

                # Zmergowano parę w tokenie
                updated_token = self._merge_pairs_in_tokens(tokens[idx], best_pair, merged_pair)

                # Wyliczamy ilośc symboli w nowym tokenie
                old_symbols = Counter(tokens[idx])
                new_symbols = Counter(updated_token)
                for symbol in old_symbols.keys() | new_symbols.keys():
                    delta = (new_symbols[symbol] - old_symbols[symbol]) * frequency[idx]

                    # Obliczamy czy coś się zmieniło
                    # Jeżeli 0 - nic się nie zmieniło
                    # Jeżeli 1<= Dodano nową parę
                    # Jeżeli 0>= Usunięto parę (nowa zastapiła część starych par)
                    if delta:
                        char_counts[symbol] += delta

                # Spróbuj to wsadzić do pairs
                old_pairs = Counter(zip(tokens[idx], tokens[idx][1:]))
                new_pairs = Counter(zip(updated_token, updated_token[1:]))
                for pair in old_pairs.keys() | new_pairs.keys():
                    # Obliczamy czy coś się zmieniło
                    # Jeżeli 0 - nic się nie zmieniło
                    # Jeżeli 1<= Dodano nową parę
                    # Jeżeli 0>= Usunięto parę (nowa zastapiła część starych par)
                    delta = (new_pairs[pair] - old_pairs[pair]) * frequency[idx]

                    # Jeżeli coś się zmieniło (nie jest 0)
                    if delta:
                        # Odejmujemy lub dodajemy ilośc wystąpień pary i symbolu w zależności od zmiany
                        pair_counts[pair] += delta
                        change_affected_pairs.add(pair)

                    # Jeżeli para występuje w new_pairs, dodajemy jej wystąpowania
                    if new_pairs[pair]:
                        pair_occurances[pair].add(idx)

                        # Dodajemy nowe symbole i jakie pary tworzą
                        symbol_to_pairs[pair[0]].add(pair)
                        symbol_to_pairs[pair[1]].add(pair)

                    # Jeżeli para nie występuje w nowych parach, to usuwamy jej miejsce występowania (bo jej już tam nie ma)
                    else:
                        pair_occurance = pair_occurances[pair]
                        if pair_occurance is not None:
                            pair_occurance.discard(idx)

                # Aktualizujemy token, dając mu jego zmergowaną wersję
                tokens[idx] = updated_token

            # Pobieramy wszystkie pary, które mają którąś z tych elementów
            affected_pairs = set(change_affected_pairs)
            for symbol in (best_pair[0], best_pair[1], merged_pair):
                affected_pairs |= set(symbol_to_pairs.get(symbol, ()))

            for pair in affected_pairs:

                # Sprawdzamy ile jest występowań pary
                count = pair_counts.get(pair, 0)

                # Jeżeli para już nie wystepuje, usuwamy ją z counts i occurances
                if count <= 0:
                    pair_counts.pop(pair, None)
                    pair_occurances.pop(pair, None)
                    scores.pop(pair, None)

                    # Usuwamy parę z bazy symoli i par
                    symbol_to_pairs[pair[0]].discard(pair)
                    symbol_to_pairs[pair[1]].discard(pair)

                else:
                    recalculate_score(pair)

            progress_bar.update(1)
        progress_bar.close()

        # budujemy drzewo do efektywniejszej tokenizacji
        self._build_trie()

    def _build_trie(self):
        self.trie = marisa_trie.Trie(self.vocabulary.tokens)
        self.cache.clear()

    def tokenize(self, batch_mapped_pre_tokens: BatchMappedPreTokens, encode: bool = False, dropout: float = 0.0) -> None:
        for mapped_pre_tokens in batch_mapped_pre_tokens:
            tokenized_tokens: list[tuple[str | int, ...]] = []
            tokenized_alignments: list[tuple[int, ...]] = []

            # single_token to pojedynczy token / słowo np. Huggingface
            for single_token, alignment in zip(mapped_pre_tokens.pre_tokens, mapped_pre_tokens.alignment):

                # Jeżeli słowo będzie za długie, np. losowy ciąg znaków, URL lub coś innego
                # Pozwala uniknąc kosztownego przechodzenia przez trie dla nietypowych długich "słów"
                if len(single_token) > self.max_input_chars_per_word:
                    chars = (self.vocabulary.special_tokens['unk_token'],)
                    continue

                # Jeżeli token był już tokenizowany, po postu wyciągamy go z cache tysamym oszczędzając czas
                # Jeżeli dropout jest włączony wyłączamy cachowanie
                chars = None if dropout > 0.0 else self.cache.get(single_token, None)
                if chars is None:

                    # Inicjalizujemy listę
                    chars: list[str] = []

                    # Tworzymy słowo '</b>𝓗𝓮𝓵𝓵𝓸'
                    symbols = self.beggining_of_token_suffix + single_token
                    char_idx, n = 0, len(symbols)
                    while char_idx < n:
                        # Zajdujemy elementy char po char (schodząc po trie) np. ['L', 'Lo', 'Lorem']
                        matching_elements = self.trie.prefixes(symbols[char_idx:])

                        # Jeżeli nie znaleźliśmy pasujących elementów
                        if matching_elements is None:
                            return (self.vocabulary.special_tokens['unk_token'],)

                        # Wybieramy ostatni - ten najdłóższy
                        best_match = matching_elements[-1]

                        # Dodajemy najlepszy (ostateczny) token do charów
                        chars.append(best_match)
                        char_idx += len(best_match)

                    chars = tuple(chars)

                    # Dodajemy token do cache (Jeżeli nie dropoutujemy)
                    if len(self.cache) < self.cache_limit and dropout == 0.0:
                        self.cache[single_token] = chars

                # Nie udało się pozyskać cache, robimy od nowa na piechotę
                # Inicjalizuje zmienną z wynikami tokenizacji
                # Inicjalizuje zmienna z alignmentem tokenu
                token_result: list[str | int, ...] = []
                token_alignment: list[tuple[int, ...]] = []

                # Obliczamy alignment sub_token po sub_tokenie ('Hug', 'g', 'in', 'g</s>')
                position = 0
                for sub_token in chars:
                    # Jeżeli chcemy od razu enkodować (zamienić na vocab id)
                    if encode:
                        token_result.append(self.vocabulary.get(sub_token))

                    # Obliczamy odpowiendnio alignment
                    sub_token_span = len(sub_token)
                    if self.beggining_of_token_suffix in sub_token:
                        sub_token_span -= len(self.beggining_of_token_suffix) * sub_token.count(
                            self.beggining_of_token_suffix)

                    token_alignment.append(tuple(alignment[position:position + sub_token_span]))
                    position += sub_token_span

                # Jeżeli nie wybrano enkodowania
                token_result = chars if not encode else tuple(token_result)

                # Doajemy do tokenów
                tokenized_tokens.append(token_result)
                tokenized_alignments.append(tuple(token_alignment))

                # Zapisujemy tokeny i aligment do obiektu
            mapped_pre_tokens.pre_tokens = tokenized_tokens
            mapped_pre_tokens.alignment = tokenized_alignments

    def encode(self, batch_mapped_pre_tokens: BatchMappedPreTokens, dropout: float = 0.0) -> BatchProcessTokens:
        self.tokenize(batch_mapped_pre_tokens, encode=True, dropout=dropout)
        return BatchProcessTokens(batch_mapped_pre_tokens)

    def save(self, name: str = None, path: str = None) -> str:
        vocabulary_data = {
            "special_tokens": self.vocabulary.special_tokens,
            "token_to_id": self.vocabulary.token_to_id,
        }

        subword_tokenizer_data = {
            "vocabulary_data": vocabulary_data,
            "max_input_chars_per_word": self.max_input_chars_per_word
        }

        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"wordpiece_{timestamp}"

        if not path:
            path = "subword_tokenizers"

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{name}.pkl")

        with open(filepath, 'wb') as f:
            pickle.dump(subword_tokenizer_data, f)

        return filepath

    @classmethod
    def load(cls, filename: str = None, path: str = 'subword_tokenizers') -> WordPiece:
        if not filename:
            raise ValueError("Subword Tokenizer filename not specified!")

        filepath = os.path.join(path, f"{filename}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'rb') as f:
            subword_tokenizer_data = pickle.load(f)

        vocabulary = Vocabulary.__new__(Vocabulary)
        vocabulary.special_tokens = subword_tokenizer_data["vocabulary_data"]["special_tokens"]
        vocabulary.token_to_id = subword_tokenizer_data["vocabulary_data"]["token_to_id"]
        vocabulary.tokens = list(vocabulary.token_to_id.keys())

        wordpiece = cls.__new__(cls)
        wordpiece.vocabulary: Vocabulary = vocabulary
        wordpiece.max_input_chars_per_word: int = subword_tokenizer_data["max_input_chars_per_word"]
        wordpiece._build_trie()

        wordpiece.cache: dict[str, tuple[tuple[str]]] = {}
        wordpiece.cache_limit: int = 1 << 20

        return wordpiece


# https://aclanthology.org/2021.emnlp-main.160.pdf
class FastWordPiece(SubwordTokenizer):
    pass


class Unigram(SubwordTokenizer):
    pass


class ProcessTokens():
    def __init__(self, mapped_pre_tokens: MappedPreTokens) -> None:
        self.tokens: list[int] = list(it.chain.from_iterable(mapped_pre_tokens.pre_tokens))
        self.attention_mask: list[int] = [1] * len(self.tokens)

    def __iter__(self):
        yield self

    def __repr__(self) -> str:
        return str(self.tokens, self.attention_mask)

    def __len__(self) -> int:
        return len(self.tokens)


class BatchProcessTokens():
    def __init__(self, batch_mapped_pre_tokens: BatchMappedPreTokens) -> None:
        self.process_tokens: list[ProcessTokens] = [ProcessTokens(mapped_pre_tokens) for mapped_pre_tokens in batch_mapped_pre_tokens]

    def __iter__(self):
        return iter(self.process_tokens)

    def __repr__(self) -> str:
        return str([single_process_tokens.tokens for single_process_tokens in self.process_tokens])

    def __len__(self) -> int:
        return len(self.process_tokens)


class PostProcessor(ABC):

    @abstractmethod
    def process(self, batch_process_tokens: BatchProcessTokens):
        pass


class TruncationPostProcessor(PostProcessor):
    """
    Przycina tablicę tokenów do ustalonej długości
    direction określa z której strony przycinać
    np. right - lewa część zostaje, a prawa zostaje ucięta
    """

    def __init__(self, length: int, direction: str = 'right') -> None:
        if direction not in ('right', 'left'):
            raise ValueError("Value of 'direction' must be one of 'right', 'left'")

        self.length = length
        self.direction = direction

    def process(self, batch_process_tokens: BatchProcessTokens) -> None:
        if self.direction == 'right':
            for process_tokens in batch_process_tokens:
                process_tokens.tokens = process_tokens.tokens[:self.length]
                process_tokens.attention_mask = process_tokens.attention_mask[:self.length]

        else:
            for process_tokens in batch_process_tokens:
                position = max(len(processed_tokens.tokens) - self.length, 0)

                process_tokens.tokens = process_tokens.tokens[position:]
                process_tokens.attention_mask = process_tokens.attention_mask[position:]


class SpecialTokensPostProcessor(PostProcessor):
    """
    Dodaje template do tokenów np. mozna z tym wykonać: <|im_start|>system<|im_sep|>You are a helpful assistant<|im_end|><|im_start|>user<|im_sep|><|im_end|><|im_start|>assistant<|im_sep|>
    """

    def __init__(self, vocabulary: Vocabulary, template: list(str) = (Vocabulary.special_tokens['bos_token'], '$',
                                                                      Vocabulary.special_tokens['eos_token'])) -> None:
        if not any(item.startswith("$") for item in template):
            raise ValueError("template must include at least one '$'")

        self.template = template
        self.vocabulary = vocabulary

    def process(self, batch_process_tokens: BatchProcessTokens):
        for process_tokens in batch_process_tokens:
            tokens = []

            for instruction in self.template:
                if instruction.startswith("$"):
                    tokens.extend(process_tokens.tokens)

                else:
                    tokens.append(self.vocabulary.get(instruction))

                if instruction not in self.vocabulary.special_tokens.values():
                    process_tokens.attention_mask.append(1)

                else:
                    process_tokens.attention_mask.append(0)

            process_tokens.tokens = tokens


class PaddingPostProcessor(PostProcessor):
    """
    Dodaje pading aby wrównać do określonej długości
    """

    def __init__(self, vocabulary: Vocabulary, length: int, at: str = 'right') -> None:
        if at not in ('right', 'left'):
            raise ValueError("Value of 'at' must be one of 'right', 'left'")

        self.at = at
        self.length = length
        self.vocabulary = vocabulary

    def process(self, batch_process_tokens: BatchProcessTokens) -> None:
        if self.at == 'right':
            for process_tokens in batch_process_tokens:
                process_tokens.tokens.extend([self.vocabulary.get(self.vocabulary.special_tokens['pad_token'])] * max(0,
                                                                                                                      self.length - len(
                                                                                                                          process_tokens.tokens)))
                process_tokens.attention_mask.extend([0] * max(0, self.length - len(process_tokens.attention_mask)))

        else:
            for process_tokens in batch_process_tokens:
                process_tokens.tokens = [self.vocabulary.get(self.vocabulary.special_tokens['pad_token'])] * max(0,
                                                                                                                 self.length - len(
                                                                                                                     process_tokens.tokens)) + processed_tokens.tokens
                process_tokens.attention_mask = [0] * max(0, self.length - len(
                    process_tokens.attention_mask)) + processed_tokens.attention_mask


class PostProcessorSequence(PostProcessor):
    """
    Pozwala połączyć wiele post processorów
    """

    def __init__(self, post_processors: list[PostProcessor]) -> None:
        self.post_processors = post_processors

    def process(self, batch_process_tokens: BatchProcessTokens) -> None:
        for post_processor in self.post_processors:
            for process_tokens in batch_process_tokens:
                post_processor.process(process_tokens)



class TokenizationPipeline(Pipeline):
    def __init__(self, text: list[str], subword_tokenizer: SubwordTokenizer = ByteLevelPreTokenizer()) -> None:
        ms = BatchMappedString(text)

        self.n_seq = NormalizationSequence(normalizers=[UnicodeNormalization()])
        mpt = self.n_seq.normalize(ms)

        self.pt_seq = PreTokenizerSequence(pre_tokenizers=[RegexPreTokenizer(regex=RegexPreTokenizer.GPT4o), subword_tokenizer])
        self.pt_seq.pre_tokenize(mpt)

        self.tokenizer = BPE()
        self.tokenizer.train(mpt)

    def tokenize(self, text: list[str]) -> BatchMappedPreTokens:
        ms = BatchMappedString(text)
        mpt = self.n_seq.normalize(ms)
        self.pt_seq.pre_tokenize(mpt)
        self.tokenizer.tokenize(mpt)

        return mpt

    def encode(self, text: list[str], max_length: int) -> BatchProcessTokens:
        ms = BatchMappedString(text)
        mpt = self.n_seq.normalize(ms)
        self.pt_seq.pre_tokenize(mpt)
        pt = self.tokenizer.encode(mpt)
        PostProcessorSequence(post_processors=[
            TruncationPostProcessor(max_length),
            SpecialTokensPostProcessor(self.tokenizer.vocabulary),
            PaddingPostProcessor(self.tokenizer.vocabulary, max_length)
        ]).process(pt)

        return pt

if __name__ == "__main__":
    pass
    # corpus = ["Lorem ipsum dolor sit amed.",
    #           "Brown fox jumps over or the lazy dog.",
    #           "Hugging Hug Hugged, Hugg Hug Huggingface Face.",
    #           "𝓗𝓮𝓵𝓵𝓸 𝓦𝓸𝓻𝓵𝓭",
    #           "怨ン縁",
    #           "ゔみバ"]
    #
    # ms = BatchMappedString(corpus)
    # mpt = NormalizationSequence(normalizers=[UnicodeNormalization()]).normalize(ms)
    # RegexPreTokenizer(RegexPreTokenizer.GPT2).pre_tokenize(mpt)
    # # PreTokenizerSequence(pre_tokenizers=[RegexPreTokenizer(), ByteLevelPreTokenizer()]).pre_tokenize(mpt)
    #
    # subword_tokenizer = WordPiece()
    # subword_tokenizer.train(mpt, 66)
    #
    # print(subword_tokenizer.vocabulary.tokens)
    # print(subword_tokenizer.trie)
    #
    # subword_tokenizer.tokenize(mpt)
    #
    # for mptt in mpt:
    #     print(mptt.pre_tokens)
    #     print(mptt.original)
    #     print(mptt.alignment, '\n')

    # # subword_tokenizer.tokenize(mpt)
    # pt = subword_tokenizer.encode(mpt)
    #
    # print(subword_tokenizer.decode(mpt))
    #
    # # pt = subword_tokenizer.encode(mpt)
    #
    # PostProcessorSequence(post_processors=[
    #     TruncationPostProcessor(6),
    #     SpecialTokensPostProcessor(subword_tokenizer.vocabulary),
    #     PaddingPostProcessor(subword_tokenizer.vocabulary, 10),
    # ]).process(pt)
    #
    # print(pt)
    #
    # print(subword_tokenizer.decode(pt))
