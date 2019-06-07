import numpy as np
import matplotlib.pyplot as plt


# データ数 (必ずクラス数の倍数)
NUM_DATA = 50
# クラス数
NUM_CLASS = 2


def main() -> None:
    X, y = genarate_dataset()
    return


def genarate_dataset(num_data: int=NUM_DATA, num_class: int=NUM_CLASS) -> (np.ndarray, np.ndarray):
    """
    データセットの作成
    @param:
        num_data: データ数 (= NUM_DATA)
        num_class: クラス数 (= NUM_CLASS)
    @return:
        X: 説明変数 (num_data, 2)
        y: 目的変数 (num_data,)
    """
    X = []
    y = []
    for b, c in zip([-15, -5], [1, -1]):
        X_i_0 = np.ones((num_data // num_class, 1))
        X_i_1 = np.random.randn(num_data // num_class, 1) + b
        X_i_2 = np.random.randn(num_data // num_class, 1)
        X_i = np.hstack((X_i_0, X_i_1, X_i_2))
        y_i = [c for _ in range(num_data // num_class)]
        X += list(X_i)
        y += y_i
    X = add_noize(np.array(X))
    return np.array(X), np.array(y)


def add_noize(X: np.ndarray) -> np.ndarray:
    """
    データにノイズを付与
    @param @return:
        X: 説明変数 (50, 2)
    """
    for i in [0, 1]:
        X[i][1] += 10
    return X


if __name__ == '__main__':
    main()
