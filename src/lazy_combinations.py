from typing import Iterable, Iterator, List, Tuple


class LazyCombinations:

    def __init__(self, data: Iterable[int], n: int):
        self.data: List[int] = list(data)
        self.n: int = n
        if n < 0:
            raise ValueError("n must be >= 0")
        if n > len(self.data):
            # Keine Kombinationen möglich
            self._empty = True
        else:
            self._empty = False

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        if self._empty:
            return

        k = self.n
        arr = self.data
        r = list(range(k))
        m = len(arr)

        yield tuple(arr[i] for i in r)

        while True:
            i = k - 1
            while i >= 0 and r[i] == i + (m - k):
                i -= 1
            if i < 0:
                break  # fertig
            r[i] += 1
            for j in range(i + 1, k):
                r[j] = r[j - 1] + 1
            yield tuple(arr[i] for i in r)
