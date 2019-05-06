import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# データ数
NUM_DATA = 100
# 更新回数
NUM_EPOCH = 1000
# x軸の最小値
X_MIN = -3.0
# x軸の最大値
X_MAX = 3.0
# ノイズ幅
NOISE_AMPLITUDE = 0.2
# 初期パラメータの最小値
PARAM_MIN = -1.0
# 初期パラメータの最大値
PARAM_MAX = 1.0
# ガウス幅
GAUSS_WIDTH = 1.0
# 正則化パラメータ
REG_PARAM = [0.001, 0.1, 10.0]


def main() -> None:
    """
    main関数
    """
    X, y = generate_data_set()
    gram_matrix = calculate_gram_matrix(X)
    for reg_param in REG_PARAM:
        weight, z, u = init_param()
        for i in range(NUM_EPOCH):
            weight, z, u = update(weight, z, u, gram_matrix, y, reg_param)
            y_pred = predict_y(gram_matrix, weight)
            err = mean_squared_error(y, y_pred)
            print('{} / {} epoch : reg_param = {} : err = {}'.format(i+1, NUM_EPOCH, reg_param, err))
        plt.plot(X, y_pred, label='λ = {}'.format(reg_param))
    plt.plot(X, true_func(X), label='true')
    plt.scatter(X, y, color='black', marker='.', label='with noise', alpha=0.7)
    plt.title('Linear Regression with L1 normalization')
    plt.legend()
    plt.show()


def generate_data_set() -> (np.ndarray, np.ndarray):
    """
    データセットを生成
    データ数はNUM_DATA
    @return:
        X 説明変数
        y 目的変数
    """
    # 説明変数を、一様分布に従ってランダム生成
    X = np.linspace(start=X_MIN, stop=X_MAX, num=NUM_DATA)
    # 標準正規分布に従うノイズ
    noise = NOISE_AMPLITUDE * np.random.randn(NUM_DATA)
    # 目的変数
    y = true_func(X) + noise
    return X, y


def true_func(x: np.ndarray) -> np.ndarray:
    """
    真の関数
    @param: x 入力ベクトル（今回は1次元）
    @return: 真の関数
    """
    return np.sin(np.pi * x) / (np.pi * x) + 0.1 * x


def gauss_kernel(a: np.ndarray, b: np.ndarray, gauss_width: float=GAUSS_WIDTH) -> float:
    """
    ガウスカーネル
    @param:
        a, b 入力ベクトル
        gauss_width ハイパーパラメータ（ガウス幅）
    @return: ガウスカーネル
    """
    return np.exp(-np.linalg.norm(a - b) ** 2 / (2 * gauss_width ** 2))


def calculate_gram_matrix(X: np.ndarray, gauss_width: float=GAUSS_WIDTH) -> np.ndarray:
    """
    グラム行列を計算
    @param:
        X 説明変数
        gauss_width ハイパーパラメータ（ガウス幅）
    @return: グラム行列
    """
    # グラム行列の初期化
    gram_matrix = np.empty((NUM_DATA, NUM_DATA))
    for i in range(NUM_DATA):
        for j in range(NUM_DATA):
            gram_matrix[i][j] = gauss_kernel(X[i], X[j], gauss_width)
    return gram_matrix


def init_param() -> (np.ndarray, np.ndarray, np.ndarray):
    """ パラメータの初期化 """
    weight = (PARAM_MAX - PARAM_MIN) * np.random.rand(NUM_DATA) + PARAM_MIN
    z = weight.copy()
    u = (PARAM_MAX - PARAM_MIN) * np.random.rand(NUM_DATA) + PARAM_MIN
    return weight, z, u


def update(weight: np.ndarray, z: np.ndarray, u: np.ndarray,
           gram_matrix: np.ndarray, y: np.ndarray, reg_param: float) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    重みベクトル、zベクトル、uベクトルを更新する
    @param:
        weight 重みベクトル
        z zベクトル
        u uベクトル
        gram_matrix グラム行列
        y 目的変数
        reg_param 正則化パラメータ
    @return
        weight_updated 更新後重みベクトル
        z 更新後zベクトル
        u 更新後uベクトル
    """
    def update_weight(gram_matrix: np.ndarray, y: np.ndarray, u:np.ndarray, z: np.ndarray) -> np.ndarray:
        """ 重みベクトルの更新 """
        A = np.dot(gram_matrix.T, gram_matrix) + np.identity(gram_matrix.shape[0])
        b = np.dot(gram_matrix.T, y) - u + z
        return np.linalg.solve(A, b)

    def update_z(weight_updated: np.ndarray, u: np.ndarray, reg_param: float) -> np.ndarray:
        """ zベクトルの更新 """
        return np.maximum(0, weight_updated + u - reg_param) + np.minimum(0, weight_updated + u + reg_param)

    def update_u(u: np.ndarray, weight_updated: np.ndarray, z_updated: np.ndarray) -> np.ndarray:
        """ uベクトルの更新 """
        return u + weight_updated - z_updated

    # 必ずこの順番で更新する
    weight_updated = update_weight(gram_matrix, y, u, z)
    z_updated = update_z(weight_updated, u, reg_param)
    u_updated = update_u(u, weight_updated, z_updated)
    return weight_updated, z_updated, u_updated


def predict_y(gram_matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    予測モデル
    @param:
        gram_matrix グラム行列
        weight 重みベクトル
    @return: 予測値
    """
    return np.dot(gram_matrix, weight)


if __name__ == '__main__':
    main()
