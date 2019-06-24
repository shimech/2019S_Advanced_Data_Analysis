import numpy as np
import matplotlib.pyplot as plt


# 訓練データ数
NUM_TRAIN = 100
# 訓練データ分布
TRAIN_RATE = 0.9
# テストデータ数
NUM_TEST = 100
# テストデータ分布
TEST_RATE = 0.1


def main() -> None:
    """ main """
    X_train, X_test, y_train, y_test = generate_dataset()
    A_pp = caluculate_A(X_train, y_train, [1, 1])
    A_pn = caluculate_A(X_train, y_train, [1, -1])
    A_nn = caluculate_A(X_train, y_train, [-1, -1])
    b_p = caluculate_b(X_train, X_test, y_train, 1)
    b_n = caluculate_b(X_train, X_test, y_train, -1)
    pi = (A_pn - A_nn - b_p + b_n) / (2 * A_pn - A_pp - A_nn)
    if pi < 0:
        pi = 0
    elif pi > 1:
        pi = 1
    else:
        pass
    print("π = {}".format(pi))
    theta_w = caluculate_theta(X_train, y_train, pi)
    theta_unw = caluculate_theta(X_train, y_train, 1.0)
    visualize(X_test, y_test, theta_w, theta_unw)
    return


def generate_dataset() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    データセットを生成
    @return:
        X_train: 訓練データの説明変数 (100, 3)
        X_test: テストデータの説明変数 (100, 3)
        y_train: 訓練データの目的変数 (100,)
        t_test: テストデータの目的変数 (100,)
    """

    def generate_variables(num: int, rate: float) -> (np.ndarray, np.ndarray):
        """
        説明変数と目的変数の生成
        @param:
            num: データ数
            rate: データ分布
        @return:
            X: 説明変数 (num, 3)
            y: 目的変数 (num,)
        """
        X_0 = np.ones((num, 1))
        X_1_group1 = np.random.randn(int(num * rate), 1) - 2.0
        X_1_group2 = np.random.randn(int(num - num * rate), 1) + 2.0
        X_1 = np.vstack((X_1_group1, X_1_group2))
        X_2 = 2.0 * np.random.randn(num, 1)
        X = np.hstack((X_0, X_1, X_2))
        y_group1 = np.ones(int(num * rate))
        y_group2 = -np.ones(int(num - num * rate))
        y = np.hstack((y_group1, y_group2))
        return X, y

    X_train, y_train = generate_variables(NUM_TRAIN, TRAIN_RATE)
    X_test, y_test = generate_variables(NUM_TEST, TEST_RATE)
    return X_train, X_test, y_train, y_test


def caluculate_A(X: np.ndarray, y: np.ndarray, labels: list) -> float:
    """
    スライド中"A"の計算
    @param:
        X: 説明変数 (100, 3)
        y: 目的変数 (100,)
        labels: 対象とするラベル (2,)
    @return:
        A: スライド中"A"
    """
    X_use = np.array([X[np.where(y == labels[0])], X[np.where(y == labels[1])]])
    nums = [len(X_use[0]), len(X_use[1])]
    sum_val = 0
    for x_1 in X_use[0]:
        for x_2 in X_use[1]:
            sum_val += np.linalg.norm(x_2 - x_1)
    A = sum_val / (nums[0] * nums[1])
    return A


def caluculate_b(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, label: int) -> float:
    """
    スライド中"b"の計算
    @param:
        X_train: 訓練データの説明変数 (100, 3)
        X_test: テストデータの説明変数 (100, 3)
        y_train: 訓練データの目的変数 (100,)
        label: 対象とするラベル
    @return:
        b: スライド中"b"
    """
    X_train_use = X_train[np.where(y_train == label)]
    nums = [len(X_train_use), len(X_test)]
    sum_val = 0
    for x_train in X_train_use:
        for x_test in X_test:
            sum_val += np.linalg.norm(x_test - x_train)
    b = sum_val / (nums[0] * nums[1])
    return b


def caluculate_theta(X: np.ndarray, y: np.ndarray, pi: float) -> np.ndarray:
    """
    最適重みベクトルを計算
    @param:
        X: 説明変数
        y: 目的変数
        pi: 重要度 (= p_test / p_train)
    @return:
        theta: 重みベクトル (3,)
    """
    if pi < 1:
        X_p = X[np.where(y == 1)]
        X_n = X[np.where(y == -1)]
        y_p = y[np.where(y == 1)]
        y_n = y[np.where(y == -1)]
    else:
        X_p = X
        X_n = X
        y_p = y
        y_n = y
    A = pi * np.dot(X_p.T, X_p) + (1 - pi) * np.dot(X_n.T, X_n)
    b = pi * np.dot(X_p.T, y_p) + (1 - pi) * np.dot(X_n.T, y_n)
    return np.linalg.solve(A, b)


def predict(x: np.ndarray, theta: np.ndarray) -> float:
    """
    予測値の計算
    @param:
        x: 説明変数 (3,)
        theta: 重みベクトル (3,)
    @return: 予測値
    """
    return np.dot(x, theta)


def visualize(X_test: np.ndarray, y_test: np.ndarray,
              theta_weighted: np.ndarray, theta_unweighted: np.ndarray) -> None:
    """
    決定境界の可視化
    @param:
        X_test: テストデータの説明変数
        y_test: テストデータの目的変数
        theta_weighted: 重要度付き重みベクトル
        theta_unweighted: 重要度なし重みベクトル
    """
    plt.scatter(X_test[np.where(y_test == 1)][:, 1], X_test[np.where(y_test == 1)][:, 2], c="blue", marker="o", alpha=0.7)
    plt.scatter(X_test[np.where(y_test == -1)][:, 1], X_test[np.where(y_test == -1)][:, 2], c="red", marker="o", alpha=0.7)
    ran = np.linspace(start=-10, stop=10, num=100)
    plt.plot(-(theta_weighted[0] + theta_weighted[2] * ran) / theta_weighted[1], ran, color="green", linewidth=3.0, label="Weighted")
    plt.plot(-(theta_unweighted[0] + theta_unweighted[2] * ran) / theta_unweighted[1], ran, color="black", linewidth=1.0, label="Unweighted", linestyle="dashed")
    plt.legend()
    print("A graph is showed.")
    plt.show()
    return


if __name__ == "__main__":
    main()
