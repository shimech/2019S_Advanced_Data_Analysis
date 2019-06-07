import numpy as np
import matplotlib.pyplot as plt


# データ数 (必ず偶数)
NUM_DATA = 200
# 学習係数
EPSILON = 0.05
# 正則化パラメータ
LAMBDA = 0.01
# エポック数
NUM_EPOCH = 1000


def main() -> None:
    X, y = generate_dataset()
    X, y = add_noise(X, y)
    weight = init_weight()
    for i in range(NUM_EPOCH):
        weight = update_weight(weight, X, y)
        print('{} / {} epoch | weight = {}'.format(i+1, NUM_EPOCH, weight))
    visualize(X, y, weight)


def generate_dataset(num_data: float=NUM_DATA) -> (np.ndarray, np.ndarray):
    """
    データセットの作成
    @param: num_data: データ数 (= NUM_DATA)
    @return:
        X: 説明変数 (200, 3)
        y: 目的変数 (200,)
    """
    X_0 = np.ones((num_data, 1))
    X_1_left = np.random.normal(loc=-5, scale=1, size=(num_data // 2, 1))
    X_1_right = np.random.normal(loc=5, scale=1, size=(num_data // 2, 1))
    X_2 = np.random.randn(num_data, 1)
    X_1 = np.vstack((X_1_left, X_1_right))
    X = np.hstack((X_0, X_1, X_2))
    y = np.array([1] * (num_data // 2) + [-1] * (num_data // 2))
    return X, y


def add_noise(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    データにノイズを付与
    @param @return:
        X: 説明変数 (200, 3)
        y: 目的変数 (200,)
    """
    X[:3, 2] -= 5
    y[:3] = -1
    X[-3:, 2] += 5
    y[-3:] = 1
    return X, y


def init_weight() -> np.ndarray:
    """ 重みベクトルの初期化 """
    return np.random.randn(3)


def update_weight(weight: np.ndarray, X: np.ndarray, y: np.ndarray,
                  e: float=EPSILON, l: float=LAMBDA) -> np.ndarray:
    """
    重みの更新
    @param:
        weight: 重みベクトル (3,)
        X: 説明変数 (200, 3)
        y: 目的変数 (200,)
        e: 学習係数 (= EPSILON)
        l: 正則化パラメータ (= LAMBDA)
    @return: 更新後重みベクトル
    """
    sum_grad = np.array([0 for _ in range(len(weight))], dtype='float')
    for i in range(len(y)):
        if 1 - predict(X, weight)[i] * y[i] > 0:
            sum_grad -= y[i] * weight
        else:
            pass
    weight_updated = weight - e * (sum_grad + l * weight)
    return weight_updated


def predict(X: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    予測値
    @param:
        X: 説明変数 (200, 3)
        weight: 重みベクトル (3,)
    @return: 予測値 (200,)
    """
    return np.dot(X, weight)


def visualize(X: np.ndarray, y: np.ndarray, weight: np.ndarray) -> None:
    """
    決定境界を表示
    @param:
        X: 説明変数 (200, 3)
        y: 目的変数 (200,)
        weight: 重みベクトル (3,)
    """
    index_blue = np.where(y == 1)
    index_red = np.where(y == -1)
    plt.scatter(X[index_blue][:, 1], X[index_blue][:, 2], c='blue', marker='o', alpha=0.7)
    plt.scatter(X[index_red][:, 1], X[index_red][:, 2], c='red', marker='o', alpha=0.7)
    plt.plot(X[:, 1], -(weight[0] + weight[1] * X[:, 1]) / weight[2], c='black')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


if __name__ == '__main__':
    main()
