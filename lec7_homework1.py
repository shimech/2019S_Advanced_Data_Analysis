import numpy as np
import matplotlib.pyplot as plt


# データ数 (必ずクラス数の倍数)
NUM_DATA = 90
# クラス数
NUM_CLASS = 3
# ガウス幅
H = 1.0
# 正則化パラメータ
LAMBDA = 1.0


def main() -> None:
    X, y = genarate_dataset()
    K = gram_matrix(X, y)
    theta = calc_theta(y, K)
    proba = predict_proba(K, theta)
    visualize(X, y, proba)


def genarate_dataset(num_data: int=NUM_DATA, num_class: int=NUM_CLASS) -> (np.ndarray, np.ndarray):
    """
    データセットの作成
    @param:
        num_data: データ数 (= NUM_DATA)
        num_class: クラス数 (= NUM_CLASS)
    @return:
        X: 説明変数 (num_data,)
        y: 目的変数 (num_data,)
    """
    X = []
    y = []
    for b, c in zip([-3, 0, 3], range(num_class)):
        X_i = np.random.randn(num_data // num_class) + b
        y_i = [c for _ in range(num_data // num_class)]
        X += list(X_i)
        y += y_i
    return np.array(X), np.array(y)


def gauss_kernel(a: np.ndarray, b: np.ndarray, h: float=H) -> float:
    """ ガウスカーネル """
    return np.exp(-np.linalg.norm(a - b) ** 2 / 2 * h ** 2)


def gram_matrix(X: np.ndarray, y: np.ndarray, num_data: int=NUM_DATA, num_class: int=NUM_CLASS) -> np.ndarray:
    """
    グラム行列
    @param:
        X: 説明変数 (num_data,)
        y: 目的変数 (num_data,)
        num_data: データ数 (= NUM_DATA)
        num_class: クラス数 (= NUM_CLASS)
    @return:
        K: グラム行列 (num_class, num_data, num_data // num_class)
    """
    num_axis = [num_class, num_data, num_data // num_class]
    K = np.empty(num_axis)
    for c in range(num_axis[0]):
        X_com = X[np.where(y == c)]
        for i in range(num_axis[1]):
            for j in range(num_axis[2]):
                K[c][i][j] = gauss_kernel(X[i], X_com[j])
    return K


def calc_theta(y: np.ndarray, K: np.ndarray, l: float=LAMBDA, num_data: int=NUM_DATA, num_class: int=NUM_CLASS) -> np.ndarray:
    """
    重みベクトルの計算
    @param:
        y: 目的変数 (num_data,)
        K: グラム行列 (num_class, num_data, num_data // num_class)
        l: 正則化パラメータ (= LAMBDA)
        num_data: データ数 (= NUM_DATA)
        num_class: クラス数 (= NUM_CLASS)
    @return:
        theta: 重みベクトル (num_class, num_data // num_class)
    """
    theta = []
    pi = np.zeros((num_class, num_data))
    for c, pi_c in enumerate(pi):
        pi_c[np.where(y == c)] = 1
        A = np.dot(K[c].T, K[c]) + l * np.eye(num_data // num_class)
        b = np.dot(K[c].T, pi_c)
        theta_c = np.linalg.solve(A, b)
        theta.append(theta_c)
    return theta


def predict_proba(K: np.ndarray, theta: np.ndarray, num_data: int=NUM_DATA, num_class: int=NUM_CLASS) -> np.ndarray:
    """
    各クラスの事後確率計算
    @param:
        K: グラム行列 (num_class, num_data, num_data // num_class)
        theta: 重みベクトル (num_class, num_data // num_class)
        num_data: データ数 (= NUM_DATA)
        num_class: クラス数 (= NUM_CLASS)
    @return:
        proba: 事後確率 (num_class, num_data)
    """
    proba = np.empty((num_class, num_data))
    for i in range(num_data):
        den = 0
        for c_ in range(num_class):
            den += max(0.0, np.dot(theta[c_], K[c_][i]))
        for c in range(num_class):
            proba[c][i] = max(0.0, np.dot(theta[c], K[c][i])) / den
    return proba


def visualize(X: np.ndarray, y: np.ndarray, proba: np.ndarray, num_data: int=NUM_DATA, num_class: int=NUM_CLASS) -> None:
    """ 予測結果の可視化 """
    colors = ['blue', 'red', 'green']
    idx = np.argsort(X)
    for c in range(num_class):
        plt.scatter(X[np.where(y == c)], np.zeros(num_data // num_class)-0.1, c=colors[c], alpha=0.7)
        plt.plot(X[idx], proba[c][idx], c=colors[c], label='p(y={}|x)'.format(c+1))
    plt.legend()
    print('A figure is showed.')
    plt.show()


if __name__ == '__main__':
    main()
