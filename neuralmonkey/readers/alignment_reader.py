from typing import List

import numpy as np

from neuralmonkey.readers.plain_text_reader import get_plain_text_reader

# tests: lint,mypy


# pylint: disable=invalid-name
def AlignmentReader(source_len, target_len, dtype=np.float32,
                    normalize=True, one_array=False):
    text_reader = get_plain_text_reader("ascii")

    def read_line(line, dest):
        for ali in line:
            i, j = map(int, ali.split("-"))
            if i < source_len and j < target_len:
                dest[j][i] = 1

        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                dest /= dest.sum(axis=1, keepdims=True)
                dest[np.isnan(dest)] = 0

    if one_array:
        def reader(files: List[str]):
            lines = list(text_reader(files))
            result = np.zeros((len(lines), target_len, source_len), dtype)

            for i, line in enumerate(lines):
                read_line(line, result[i])

            return result
    else:
        def reader(files: List[str]):
            for line in text_reader(files):
                a = np.zeros((target_len, source_len), dtype)
                read_line(line, a)

                yield a

    return reader
