import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# データ数
NUM_DATA = 1000


def main() -> None:
    """
    main関数
    """
    X = generate_dataset()
    W = similarity_matrix(X)
    X_ = projection_matrix(W)
    visualize(X_)
    return


def generate_dataset() -> np.ndarray:
    """
    データセットの生成
    @return: 説明変数 (1000, 3)
    """
    a = 3.0 * np.pi * np.random.rand(NUM_DATA)
    X = np.array([a * np.cos(a), 30.0 * np.random.rand(NUM_DATA), a * np.sin(a)]).T
    return X


def similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    類似度行列の計算
    @param:
        X: 説明変数 (1000, 3)
    @return:
        sim_mat: 類似度行列 (1000, 1000)
    """
    def distance_matrix(X: np.ndarray) -> np.ndarray:
        """
        2点間の距離の計算
        @param:
            X: 説明変数 (1000, 3)
        @return:
            dis_mat: 距離行列 (1000, 1000)
        """
        dis_mat = np.empty((NUM_DATA, NUM_DATA))
        for i in range(dis_mat.shape[0]):
            for j in range(dis_mat.shape[1]):
                dis_mat[i][j] = np.linalg.norm(X[i] - X[j]) if i != j else float("inf")
        return dis_mat
    sim_mat = np.zeros((NUM_DATA, NUM_DATA))
    dis_mat = distance_matrix(X)
    min_idxs = np.argmin(dis_mat, axis=1)
    for i, min_idx in enumerate(min_idxs):
        sim_mat[i][min_idx] = 1
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i][j] == 1:
                sim_mat[j][i] = 1
    return sim_mat


def projection_matrix(W: np.ndarray) -> np.ndarray:
    """
    埋め込み先の計算
    @param:
        W: 類似度行列 (1000, 1000)
    @return:
        proj_mat: 埋め込み先 (1000, 2)
    """
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    eig_vals, eig_vecs = scipy.linalg.eig(L, D)
    idxs_sorted = np.argsort(eig_vals)
    proj_mat = eig_vecs[:, idxs_sorted[1:3]]
    return proj_mat


def visualize(X_: np.ndarray) -> None:
    """
    可視化
    @param:
        X_: 埋め込み先説明変数 (1000, 2)
    """
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(X[:, 0], X[:, 1], X[:, 2], "o")
    plt.scatter(X_[:, 0], X_[:, 1], marker="o")
    print("A graph is showed.")
    plt.show()
    return


if __name__ == "__main__":
    main()
