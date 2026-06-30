import unicodedata

class MappedString():
    def __init__(self, text: str):
        self.alignment = list(range(len(text)))
        self.original = text
        self.text = text

    def __call__(self, start: int, end: int) -> tuple[int, int] | None:
        if end - 1 >= len(self.alignment) or end - 1 < 0:
            return None

        o_start = self.alignment[start]
        o_end = self.alignment[end - 1] + 1 if end > start else self.alignment[start]

        return (o_start, o_end)

    def __len__(self):
        return len(self.text)

class Normalization():
    def normalize(self, mapped_string: MappedString) -> None:
        pass

class LowerCaseNormalization(Normalization):
    def normalize(self, mapped_string: MappedString) -> None:
        mapped_string.text = mapped_string.text.lower()

class WhiteSpaceNormalization(Normalization):
    def normalize(self, mapped_string: MappedString) -> None:
        if len(mapped_string.text) == 0:
            return

        start = 0
        end = len(mapped_string.text)
        while start < end and mapped_string.text[start].isspace():
            start += 1

        while end > start and mapped_string.text[end - 1].isspace():
            end -= 1

        chars = []
        alignment = []
        previous_space = False
        for i in range(start, end):
            char = mapped_string.text[i]
            is_space = char.isspace()

            if is_space and previous_space:
                continue

            chars.append(char)
            alignment.append(mapped_string.alignment[i])
            previous_space = is_space

        mapped_string.text = "".join(chars)
        mapped_string.alignment = alignment

class UnicodeNormalization(Normalization):
    def __init__(self, mode: str = "NFC"):
        if mode not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError("Value of \'mode\' must be one of: NFC, NFD, NFK, NFKD")

        self.mode = mode

    def normalize(self, mapped_string: MappedString) -> None:
        n_text = unicodedata.normalize(self.mode, mapped_string.text)

        if len(n_text) == len(mapped_string.text):
            mapped_string.text = n_text
            return

        n_text = ''
        n_alignment = []
        common = 0
        for i in range(len(mapped_string.text)):
            text_window = mapped_string.text[:i + 1]
            n_text_window = unicodedata.normalize(self.mode, text_window)

            while (common < min(len(n_text), len(n_text_window))
                   and n_text[common] == n_text_window[common]):
                common += 1

            new_chars = n_text_window[common:]
            n_alignment.extend([i] * len(new_chars))
            n_text = n_text_window

        mapped_string.text = n_text
        mapped_string.alignment = n_alignment

class AccentsNormalization(Normalization):
    def normalize(self, mapped_string: MappedString) -> None:
        alignment: list[int] = []
        text: list[str] = []
        for i, char in enumerate(mapped_string.text):
            if unicodedata.category(char) == "Mn":
                continue

            text.append(char)
            alignment.append(mapped_string.alignment[i])

        mapped_string.text = ''.join(text)
        mapped_string.alignment = alignment

class ReplaceNormalization(Normalization):
    def __init__(self, target: str, replacement: str):
        if len(target) == 0:
            raise ValueError("\'target\' nie może być pustym stringiem")

        self.replacement = replacement
        self.target = target

    def normalize(self, mapped_string: MappedString) -> None:
        text = []
        alignment = []
        position = 0
        
        while True:
            i = mapped_string.text.find(self.target, posposition)
            if i == -1:
                new_text_parts.extend(mapped_string.text[posposition:])
                new_alignment.extend(mapped_string.alignment[posposition:])
                break

            new_text_parts.extend(mapped_string.text[posposition:i])
            new_alignment.extend(mapped_string.alignment[posposition:i])

            last_origin = mapped_string.alignment[i + len(self.target) - 1]
            new_text_parts.extend(self.replacement)
            new_alignment.extend([last_origin] * len(self.replacement))

            posposition = i + len(self.target)

        mapped_string.text = "".join(new_text_parts)
        mapped_string.alignment = new_alignment

class NormalizationSequence(Normalization):
    def __init__(self, normalizers: list[Normalization]) -> None:
        self.normalizers = normalizers

    def normalize(self, mapped_string: MappedString) -> None:
        for normalizer in self.normalizers:
            normalizer.normalize(mapped_string)
