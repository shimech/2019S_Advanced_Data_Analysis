import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


# データ数
NUM_DATA = 100


def main() -> None:
    """ main """
    X, y = generate_dataset()
    S_b_0, S_w_0 = calculate_scattering_matrix(X[0], y[0])
    S_b_1, S_w_1 = calculate_scattering_matrix(X[1], y[1])
    S_b = np.array([S_b_0, S_b_1])
    S_w = np.array([S_w_0, S_w_1])
    eig_vec_0 = calculate_1st_projection_vector(S_b[0], S_w[0])
    eig_vec_1 = calculate_1st_projection_vector(S_b[1], S_w[1])
    eig_vecs = np.array([eig_vec_0, eig_vec_1])
    visualize(X, y, eig_vecs)
    return


def generate_dataset() -> (np.ndarray, np.ndarray):
    """
    データセットを生成
    @return:
        X: 説明変数 (2, 100, 2)
        y: 目的変数 (2, 100)
    """
    def centerize(X: np.ndarray) -> np.ndarray:
        """
        中心化
        @param:
            X: 中心化前説明変数 (100, 2)
        @return: 中心化後説明変数 (100, 2)
        """
        X_avg = np.mean(X, axis=0)
        return X - X_avg

    X0_group0_pattern0 = np.random.randn(NUM_DATA // 2, 1) - 4
    X0_group1_pattern0 = np.random.randn(NUM_DATA // 2, 1) + 4
    X0_pattern0 = np.vstack((X0_group0_pattern0, X0_group1_pattern0))
    X1_pattern0 = np.random.randn(NUM_DATA, 1)
    X_pattern0 = np.hstack((X0_pattern0, X1_pattern0))
    X_pattern0 = centerize(X_pattern0)
    X0_group0_0_pattern1 = np.random.randn(NUM_DATA // 4, 1) - 4
    X0_group0_1_pattern1 = np.random.randn(NUM_DATA // 4, 1) + 4
    X0_group1_pattern1 = np.random.randn(NUM_DATA // 2, 1)
    X0_pattern1 = np.vstack((X0_group0_0_pattern1, X0_group1_pattern1, X0_group0_1_pattern1))
    X1_pattern1 = np.random.randn(NUM_DATA, 1)
    X_pattern1 = np.hstack((X0_pattern1, X1_pattern1))
    X_pattern1 = centerize(X_pattern1)
    X = np.array([X_pattern0, X_pattern1])
    y_pattern0 = [-1 for _ in range(NUM_DATA // 2)] + [1 for _ in range(NUM_DATA // 2)]
    y_pattern1 = [-1 for _ in range(NUM_DATA // 4)] + [1 for _ in range(NUM_DATA // 2)] + [-1 for _ in range(NUM_DATA // 4)]
    y = np.array([y_pattern0, y_pattern1])
    return X, y


def calculate_scattering_matrix(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    クラス間散布行列、クラス内散布行列の計算
    @param:
        X: 説明変数 (100, 2)
        y: 目的変数 (100,)
    @return:
        S_b: クラス間散布行列 (2, 2)
        S_w: クラス内散布行列 (2, 2)
    """
    labels = set(y)
    S_b = np.zeros((X.shape[1], X.shape[1]))
    S_w = np.zeros((X.shape[1], X.shape[1]))
    for l in labels:
        num = len(X[np.where(y == l)])
        X_mean = np.mean(X[np.where(y == l)], axis=0).reshape(-1, 1)
        S_b += num * np.dot(X_mean, X_mean.T)
        for x in X[np.where(y == l)]:
            S_w += np.dot((x - X_mean), (x - X_mean).T)
    return S_b, S_w


def calculate_1st_projection_vector(S_b: np.ndarray, S_w: np.ndarray) -> np.ndarray:
    """
    第一射影ベクトルを計算
    @param:
        S_b: クラス間散布行列 (2, 2)
        S_w: クラス内散布行列 (2, 2)
    @return: 第一射影ベクトル (2,)
    """
    A = S_b
    B = S_w
    eig_vals, eig_vecs = scipy.linalg.eig(A, B)
    max_idx = np.argmax(eig_vals)
    return eig_vecs[:, max_idx]


def visualize(X: np.ndarray, y:np.ndarray, eig_vecs: np.ndarray) -> None:
    """
    第一射影軸の可視化
    @param:
        X: 説明変数 (2, 100, 2)
        y: 目的変数 (2, 100)
        eig_vecs: 第一射影ベクトル (2, 2)
    """
    ran = np.linspace(start=-7, stop=7, num=100)
    plt.subplot(1, 2, 1)
    plt.scatter(X[0][np.where(y[0] == 1)][:, 0], X[0][np.where(y[0] == 1)][:, 1], c="blue")
    plt.scatter(X[0][np.where(y[0] == -1)][:, 0], X[0][np.where(y[0] == -1)][:, 1], c="red")
    plt.plot(ran, ran * eig_vecs[0][1] / eig_vecs[0][0], color="black")
    plt.subplot(1, 2, 2)
    plt.scatter(X[1][np.where(y[1] == 1)][:, 0], X[1][np.where(y[1] == 1)][:, 1], c="blue")
    plt.scatter(X[1][np.where(y[1] == -1)][:, 0], X[1][np.where(y[1] == -1)][:, 1], c="red")
    plt.plot(ran, ran * eig_vecs[1][1] / eig_vecs[1][0], color="black")
    print("A graph is showed.")
    plt.show()
    return


if __name__ == "__main__":
    main()
