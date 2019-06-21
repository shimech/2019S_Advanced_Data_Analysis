import itertools
import numpy as np
import matplotlib.pyplot as plt


# データ数 (必ず偶数)
NUM_DATA = 200
# テストデータ数
NUM_DATA_TEST = 100
# L2正則化パラメータ
LAMBDA = 1.0
# ラプラス正則化パラメータ
NU = 1.0
# ガウス幅
H = 1.0


def main() -> None:
    """ main関数 """
    X, y = generate_dataset()
    phi = design_matrix(X, X)
    phi_tilde = design_matrix(X[np.where(y != 0)], X)
    theta = calculate_theta(phi, phi_tilde, y[np.where(y != 0)])
    visualize(X, y, theta)
    return


def generate_dataset() -> (np.ndarray, np.ndarray):
    """
    データセットを生成
    @return:
        X: 説明変数 (200, 2)
        y: 目的変数 (200,)
    """
    inputs = np.linspace(start=0.0, stop=np.pi, num=NUM_DATA // 2)
    X1_group1 = -10.0 * (np.cos(inputs) + 0.5) + np.random.randn(NUM_DATA // 2)
    X1_group2 = -10.0 * (np.cos(inputs) - 0.5) + np.random.randn(NUM_DATA // 2)
    X2_group1 = 10.0 * np.sin(inputs) + np.random.randn(NUM_DATA // 2)
    X2_group2 = -10.0 * np.sin(inputs) + np.random.randn(NUM_DATA // 2)
    X1 = np.hstack((X1_group1, X1_group2)).reshape(-1, 1)
    X2 = np.hstack((X2_group1, X2_group2)).reshape(-1, 1)
    X = np.hstack((X1, X2))
    y = np.zeros(NUM_DATA)
    y[0], y[-1] = -1, 1
    X_labeled, y_labeled = X[np.where(y != 0)], y[np.where(y != 0)]
    X_unlabeled, y_unlabeled = X[np.where(y == 0)], y[np.where(y == 0)]
    X = np.vstack((X_labeled, X_unlabeled))
    y = np.hstack((y_labeled, y_unlabeled))
    return X, y


def gauss_kernel(a: np.ndarray, b: np.ndarray) -> float:
    """
    ガウスカーネル
    @param:
        a, b: 説明変数
    @return: ガウスカーネル
    """
    return np.exp(-np.linalg.norm(a - b) ** 2 / 2 * H ** 2)


def design_matrix(X: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    """
    計画行列
    @param:
        X: 説明変数
    @return:
        dmat: 計画行列
    """
    dmat = np.empty((len(X), NUM_DATA))
    for i in range(dmat.shape[0]):
        for j in range(dmat.shape[1]):
            dmat[i][j] = gauss_kernel(X[i], X_all[j])
    return dmat


def calculate_theta(phi: np.ndarray, phi_tilde: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    θを解析的に求める
    @param:
        phi: 全データの計画行列
        phi_tilde: ラベル付きデータのみの計画行列
        y: 目的変数
    @return: θの解析解
    """
    W = phi
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    A = np.dot(phi_tilde.T, phi_tilde) + LAMBDA * np.eye(NUM_DATA) + 2 * NU * np.dot(phi.T, np.dot(L, phi))
    b = np.dot(phi_tilde.T, y)
    return np.linalg.solve(A, b)


def predict(X_train: np.ndarray, X_test: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    予測値の計算
    @param:
        X_train: 訓練データの説明変数
        X_test: テストデータの説明変数
        theta: 重みベクトル
    @return: 予測値
    """
    K = design_matrix(X_test, X_train)
    y_pred = np.dot(K, theta)
    return np.sign(y_pred)


def visualize(X_train: np.ndarray, y: np.ndarray, theta: np.ndarray) -> None:
    """
    決定境界の可視化
    @param:
        X_train: 訓練データの説明変数
        y: 訓練データの目的変数
        theta: 重みベクトル
    """
    ran = np.linspace(start=-20, stop=20, num=NUM_DATA_TEST)
    X_test = np.array(list(itertools.product(ran, repeat=2)))
    y_pred = predict(X_train, X_test, theta)
    plt.scatter(X_train[:, 0], X_train[:, 1], s=10, c='black')
    plt.plot(X_test[np.where(y_pred == -1)][:, 0], X_test[np.where(y_pred == -1)][:, 1], color='blue', alpha=0.05, marker='o', markersize=10)
    plt.plot(X_test[np.where(y_pred == 1)][:, 0], X_test[np.where(y_pred == 1)][:, 1], color='red', alpha=0.05, marker='o', markersize=10)
    print("A graph is showed.")
    plt.show()
    return


if __name__ == "__main__":
    main()
