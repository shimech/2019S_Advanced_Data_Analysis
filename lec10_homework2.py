import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


# データ数
NUM_DATA = 100


def main() -> None:
    """ main """
    X = generate_dataset()
    W_0 = calculate_similarity_matrix(X[0])
    W_1 = calculate_similarity_matrix(X[1])
    W = np.array([W_0, W_1])
    eig_vec_0 = calculate_1st_projection_vector(X[0], W[0])
    eig_vec_1 = calculate_1st_projection_vector(X[1], W[1])
    eig_vecs = np.array([eig_vec_0, eig_vec_1])
    visualize(X, eig_vecs)
    return


def generate_dataset() -> np.ndarray:
    """
    データセットを生成
    @return:
        X: 説明変数 (2, 100, 2)
    """
    X0_pattern0 = 2 * np.random.randn(NUM_DATA, 1)
    X1_pattern0 = np.random.randn(NUM_DATA, 1)
    X_pattern0 = np.hstack((X0_pattern0, X1_pattern0))
    X0_pattern1 = 2 * np.random.randn(NUM_DATA, 1)
    X1_pattern1 = 2 * np.random.rand(NUM_DATA, 1).round() - 1 + np.random.randn(NUM_DATA, 1) / 3
    X_pattern1 = np.hstack((X0_pattern1, X1_pattern1))
    X = np.array([X_pattern0, X_pattern1])
    return X


def calculate_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    類似度行列の計算
    @param:
        X: 説明変数 (100, 2)
    @return:
        simmat: 類似度行列 (100, 100)
    """
    simmat = np.empty((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            simmat[i][j] = np.exp(-np.linalg.norm(X[i] - X[j]) ** 2)
    return simmat


def calculate_1st_projection_vector(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    第一射影ベクトルを計算
    @param:
        X: 説明変数 (100, 2)
        W: 類似度行列 (100, 100)
    @return: 第一射影ベクトル (2,)
    """
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    A = np.dot(X.T, np.dot(L, X))
    B = np.dot(X.T, np.dot(D, X))
    eig_vals, eig_vecs = scipy.linalg.eig(A, B)
    min_idx = np.argmin(eig_vals)
    return eig_vecs[:, min_idx]


def visualize(X: np.ndarray, eig_vecs: np.ndarray) -> None:
    """
    第一射影軸の可視化
    @param:
        X: 説明変数
        eig_vecs: 第一射影ベクトル
    """
    ran = np.linspace(start=-5, stop=5, num=100)
    plt.subplot(1, 2, 1)
    plt.scatter(X[0, :, 0], X[0, :, 1], c="blue")
    plt.plot(ran, ran * eig_vecs[0][1] / eig_vecs[0][0], color="black")
    plt.subplot(1, 2, 2)
    plt.scatter(X[1, :, 0], X[1, :, 1], c="red")
    plt.plot(ran, ran * eig_vecs[1][1] / eig_vecs[1][0], color="black")
    print("A graph is showed.")
    plt.show()
    return


if __name__ == "__main__":
    main()
