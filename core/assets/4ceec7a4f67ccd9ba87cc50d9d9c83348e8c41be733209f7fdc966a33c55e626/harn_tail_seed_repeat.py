"""One predeclared visual seed repeat; Skeleton seeds remain unchanged."""

from cuhkx.harn_tail import TailClassifier


class SeedRepeatTailClassifier(TailClassifier):
    def __init__(self, settings, seed):
        if seed != 20262905:
            raise ValueError("repeat expects the original fixed visual seed")
        super().__init__(settings, seed + 1)

