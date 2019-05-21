import numpy as np


NUM_DATA = 200


def main() -> None:


def generate_dataset(num_data: float=NUM_DATA) -> (np.ndarray, np.ndarray):
    X = np.random.randn(num_data, 2)
    X[:, 0] += 5
    index_left = np.random.choice(np.arange(num_data), num_data // 2, replace=False)
    X[:][index_left] -= 10


def update_weight(weight: np.ndarray, X: np.ndarray, y: np.ndarray,
                  e: float=EPSILON, l: float=LAMBDA) -> np.ndarray:


if __name__ == '__main__':
    main()
