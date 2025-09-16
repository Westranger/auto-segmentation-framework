from typing import List, Tuple


class DummyProductEvaluator:
    def evaluate(self, input_size: int, lst_tuples: List[Tuple[int, int]]) -> float:
        prod = input_size
        for ks, ch in lst_tuples:
            prod *= ks
            prod *= ch
        return float(prod)
