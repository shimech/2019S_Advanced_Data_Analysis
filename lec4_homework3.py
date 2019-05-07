import numpy as np
import matplotlib.pyplot as plt


# データ数
NUM_DATA = 10
# x軸の最小値
X_MIN = -3.0
# x軸の最大値
X_MAX = 3.0
# ノイズの大きさ
NOISE_AMPLITUDE = 0.2
# 外れ値生成する説明変数
INDEX_OUT = [2, 8, 9]
# 外れ値の値
OUT_VALUE = -4.0
# η (ハイパーパラメータ)
ETA = 1.0
# 初期重みベクトルの最小値
THETA_INIT_MIN = -1.0
# 初期重みベクトルの最大値
THETA_INIT_MAX = 1.0
# 繰り返し回数
NUM_EPOCH = 100


def main() -> None:
    """ main """
    X, y = generate_data_set()
    phi = caluculate_phi(X)
    theta = init_theta()
    for i in range(NUM_EPOCH):
        y_pred = predict_y(X, theta)
        w = caluculate_w(y, y_pred)
        err = loss(y, y_pred)
        theta = update_theta(phi, y, w)
        print('{} / {} epoch : theta = {} : err = {}'.format(i+1, NUM_EPOCH, theta, err))
    visualize(X, y, y_pred)


def true_func(X: np.ndarray) -> np.ndarray:
    """
    真の関数
    @param: X 説明変数
    @return: 目的変数
    """
    return X


def generate_outlier(y: np.ndarray, index_out: list=INDEX_OUT, out_value: float=OUT_VALUE) -> np.ndarray:
    """
    外れ値を生成する
    @param
        index_out: 外れ値を生成する説明変数
        y: 外れ値生成前の目的変数
        out_value: 外れ値の値 (= OUT_VALUE)
    @return 外れ値生成後の目的変数
    """
    for i in index_out:
        y[i] = out_value
    return y


def generate_data_set(x_min: float=X_MIN, x_max: float=X_MAX, num_data: int=NUM_DATA,
                      noise_amplitude: float=NOISE_AMPLITUDE, index_out: list=INDEX_OUT) -> (np.ndarray, np.ndarray):
    """
    データセットを生成する
    @param:
        x_min 説明変数の最小値 (= X_MIN)
        x_max 説明変数の最大値 (= X_MAX)
        num_data データ数 (= NUM_DATA)
        noise_amplitude ノイズの大きさ (= NOISE_AMPLITUDE)
    @return:
        X 説明変数
        y 目的変数
    """
    X = np.linspace(start=x_min, stop=x_max, num=num_data)
    noise = noise_amplitude * np.random.randn(num_data)
    y = true_func(X) + noise
    y = generate_outlier(y)
    return X, y


def caluculate_phi(X: np.ndarray) -> np.ndarray:
    """
    Φを計算
    @param: 説明変数
    @return: Φ
    """
    X = X.reshape(-1, 1)
    bias = np.ones((X.shape[0], 1))
    return np.hstack((bias, X))


def init_theta(init_min: float=THETA_INIT_MIN, init_max: float=THETA_INIT_MAX) -> np.ndarray:
    """ θの初期化 """
    return (init_max - init_min) * np.random.rand(2) + init_min


def predict_y(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    予測する
    @param:
        X 説明変数
        theta 重みベクトル
    @return 予測値 (= θ_0 + θ_1 * x)
    """
    return theta[0] + theta[1] * X


def loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """ 損失関数 (平均二乗誤差) """
    return np.linalg.norm(y - y_pred) ** 2


def caluculate_w(y: np.ndarray, y_pred: np.ndarray, eta: float=ETA) -> np.ndarray:
    """
    テューキー重みを計算
    @param:
        y 目的関数
        y_pred 予測値
        eta η (ハイパーパラメータ)
    @return テューキー重み
    """
    w = []
    for r in abs(y - y_pred):
        if r <= eta:
            w.append((1.0 - (r / eta) ** 2) ** 2)
        else:
            w.append(0)
    return np.array(w)


def update_theta(phi: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ θの更新 """
    w = np.diag(w)
    A = np.dot(np.dot(phi.T, w), phi)
    b = np.dot(np.dot(phi.T, w), y)
    return np.linalg.solve(A, b)


def visualize(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
    """ 予測結果の可視化 """
    plt.plot(X, y_pred, label='predict', color='green')
    plt.scatter(X, y, label='observe', color='blue')
    plt.title('Tukey Regression')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
