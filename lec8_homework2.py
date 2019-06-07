import numpy as np
import matplotlib.pyplot as plt


# データ数 (必ずクラス数の倍数)
NUM_DATA = 50
# クラス数
NUM_CLASS = 2
# エポック数
NUM_EPOCH = 100
# パラメータの変化率
GAMMA = 0.1


def main() -> None:
    X, y = genarate_dataset()
    mu, sigma = init_params(X)
    for epoch in range(NUM_EPOCH):
        for i in range(NUM_DATA):
            mu, sigma = update_params(mu, sigma, X[i], y[i])
        print('{} / {} epoch | μ = {}, Σ = {}'.format(epoch+1, NUM_EPOCH, mu, sigma))
    theta = np.random.multivariate_normal(mean=mu, cov=sigma)
    visualize(X, y, theta)
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
    index = np.random.permutation(np.arange(num_data))
    y = np.array(y)
    return X[index], y[index]


def add_noize(X: np.ndarray) -> np.ndarray:
    """
    データにノイズを付与
    @param @return:
        X: 説明変数 (50, 2)
    """
    for i in [0, 1]:
        X[i][1] += 10
    return X


def init_params(X: np.ndarray) -> (np.ndarray, np.ndarray):
    """ パラメータの初期化 """
    return np.random.normal(loc=0, scale=0.1, size=len(X[0])), np.random.normal(loc=0, scale=0.1, size=(len(X[0]), len(X[0])))


def update_params(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray, y: np.ndarray, gamma: float=GAMMA) -> (np.ndarray, np.ndarray):
    """
    パラメータの更新
    @param:
        mu: 期待値
        sigma: 分散
        x: 説明変数のサンプル (3,)
        y: 目的変数のサンプル (∈ {+1, -1})
        gamma: ハイパーパラメータ (= GAMMA)
    @return:
        mu: 更新後期待値
        sigma: 更新後分散
    """
    def update_mu(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
        """ 期待値の更新 """
        return mu + (y * max(0, 1 - np.dot(mu, x)) * y) * np.dot(sigma, x) / (np.dot(x, np.dot(sigma, x)) + gamma)

    def update_sigma(sigma: np.ndarray, x: np.ndarray, gamma: float) -> np.ndarray:
        """ 分散の更新 """
        return sigma - np.dot(np.dot(sigma, np.dot(x.reshape(-1, 1), x.reshape(1, -1))), sigma) / (np.dot(x, np.dot(sigma, x)) + gamma)

    return update_mu(mu, sigma, x, y, gamma) , update_sigma(sigma, x, gamma)


def visualize(X: np.ndarray, y: np.ndarray, theta: np.ndarray, num_data: int=NUM_DATA) -> None:
    """
    決定境界を表示
    @param:
        X: 説明変数 (num_data, 3)
        y: 目的変数 (num_data,)
        theta: 重みベクトル (3,)
        num_data: データ数 (= NUM_DATA)
    """
    colors = ['blue', 'red']
    x_2_sample = np.linspace(start=-2, stop=2, num=10)
    for i, c in enumerate([1, -1]):
        plt.scatter(X[:, 1][np.where(y == c)], X[:, 2][np.where(y == c)], c=colors[i], marker='o', alpha=0.7)
        plt.plot(-(theta[0] + theta[2] * x_2_sample) / theta[1], x_2_sample, c='black')
    plt.show()
    return


if __name__ == '__main__':
    main()
