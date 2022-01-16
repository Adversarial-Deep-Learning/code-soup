from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class Classifier(ABC):  # no pragma: no cover
    def __init__(self):
        pass

    @abstractmethod
    def get_prob(input_: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def get_pred(input_: List[str]) -> np.ndarray:
        pass

    def get_grad(input_: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        pass
