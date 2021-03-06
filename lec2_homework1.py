import itertools
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# データ数
NUM_DATA = 100
# 交差検証の回数
NUM_CROSS_VALIDATION = 10
# 1グループのデータ数
NUM_DATA_PER_GROUP = int(NUM_DATA / NUM_CROSS_VALIDATION)
# x軸の最小値
X_MIN = -3
# x軸の最大値
X_MAX = 3
# ノイズ幅
NOISE_AMPLITUDE = 0.2
# ガウス幅のリスト
GAUSS_WIDTH = [0.1, 1.0, 10.0]
# 正則化パラメータのリスト
REG_PARAM = [0.0001, 0.1, 10.0]


def main() -> None:
    """
    main関数
    """
    X, y = generate_data_set()
    best_param = grid_search(X, y)
    print('Best parameters are {}'.format(best_param))


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


def train_test_split(X: np.ndarray, y: np.ndarray, index_test: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    訓練データとテストデータに分割する
    @param:
        X 説明変数（交差検証回数 × 1グループのデータ数）
        y 目的変数
        index_test テストデータのインデクス
    @return:
        X_train 訓練データの説明変数
        X_test テストデータの説明変数
        y_train 訓練データの説明変数
        y_test テストデータの説明変数
    """
    X_test = X[index_test]
    y_test = y[index_test]
    X_train = np.delete(X, obj=index_test, axis=0).reshape(-1,)
    y_train = np.delete(y, obj=index_test, axis=0).reshape(-1,)
    return X_train, X_test, y_train, y_test


def true_func(x: np.ndarray) -> np.ndarray:
    """
    真の関数
    @param: x 入力ベクトル（今回は1次元）
    @return: 真の関数
    """
    return np.sin(np.pi * x) / (np.pi * x) + 0.1 * x


def gauss_kernel(a: np.ndarray, b: np.ndarray, gauss_width: float) -> float:
    """
    ガウスカーネル
    @param:
        a, b 入力ベクトル
        gauss_width ハイパーパラメータ（ガウス幅）
    @return: ガウスカーネル
    """
    return np.exp(-np.linalg.norm(a - b) ** 2 / (2 * gauss_width ** 2))


def calculate_gram_matrix(X_train: np.ndarray, gauss_width: float) -> np.ndarray:
    """
    グラム行列を計算
    @param:
        X_train 訓練データの説明変数
        gauss_width ハイパーパラメータ（ガウス幅）
    @return: グラム行列
    """
    num_train_data = len(X_train)
    # グラム行列の初期化
    gram_matrix = np.empty((num_train_data, num_train_data))
    for i in range(num_train_data):
        for j in range(num_train_data):
            gram_matrix[i][j] = gauss_kernel(X_train[i], X_train[j], gauss_width)
    return gram_matrix


def calculate_weight(gram_matrix: np.ndarray, reg_param: float, y_train: np.ndarray) -> np.ndarray:
    """
    重みベクトルを計算
    @param:
        gram_matrix グラム行列
        reg_param ハイパーパラメータ（正則化パラメータ）
        y_train 訓練データの目的変数
    @return: 重みベクトル
    """
    num_train_data = len(y_train)
    inv_mat = np.dot(gram_matrix, gram_matrix) + reg_param * np.identity(num_train_data)
    inv_mat = np.linalg.inv(inv_mat)
    coef = np.dot(inv_mat, gram_matrix.T)
    weight = np.dot(coef, y_train)
    return weight


def calculate_kernel(X_train: np.ndarray, X_test: np.ndarray, gauss_width: float) -> np.ndarray:
    """
    カーネルベクトルを計算
    @param:
        X_train 訓練データの説明変数
        X_test テストデータの説明変数
        gauss_width ハイパーパラメータ（ガウス幅）
    @return カーネルベクトル
    """
    num_train_data = len(X_train)
    num_test_data = len(X_test)
    kernel = np.empty((num_test_data, num_train_data))
    for i in range(num_test_data):
        for j in range(num_train_data):
            kernel[i][j] = gauss_kernel(X_test[i], X_train[j], gauss_width)
    return kernel


def cross_validation(X: np.ndarray, y: np.ndarray, gauss_width: float, reg_param: float) -> float:
    """
    交差検証
    @param:
        X 説明変数（横ベクトル）
        y 目的変数
        gauss_width ハイパーパラメータ（ガウス幅）
        reg_param ハイパーパラメータ（正則化パラメータ）
    @return: 平均誤差
    """
    X = X.reshape(NUM_CROSS_VALIDATION, NUM_DATA_PER_GROUP)
    y = y.reshape(NUM_CROSS_VALIDATION, NUM_DATA_PER_GROUP)
    errs = []
    for i in range(NUM_CROSS_VALIDATION):
        X_train, X_test, y_train, y_test = train_test_split(X, y, i)
        gram_matrix = calculate_gram_matrix(X_train, gauss_width)
        weight = calculate_weight(gram_matrix, reg_param, y_train)
        kernel = calculate_kernel(X_train, X_test, gauss_width)
        y_pred = np.dot(kernel, weight)
        err = mean_squared_error(y_test, y_pred)
        errs.append(err)
        print('{} / {} finished'.format(i+1, NUM_CROSS_VALIDATION))
    ave_err = np.mean(errs)
    print('err = {}'.format(ave_err))
    return ave_err


def predict(X: np.ndarray, y: np.ndarray, gauss_width: float, reg_param:float) -> np.ndarray:
    """
    予測モデル
    @param:
        X 説明変数
        y 目的変数
        gauss_width ガウス幅
        reg_param 正則化パラメータ
    @return: 予測値
    """
    gram_matrix = calculate_gram_matrix(X, gauss_width)
    weight = calculate_weight(gram_matrix, reg_param, y)
    return np.dot(gram_matrix, weight)


def visualize(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, gauss_width: float, reg_param: float, err: float, i: int) -> None:
    """
    予測結果を可視化する
    @param:
        X 説明変数
        y 目的変数
        y_pred 予測値
        gauss_width ガウス幅
        reg_param 正則化パラメータ
        err 予測値の平均二乗誤差
        i グラフのインデクス
    """
    plt.subplot(3, 3, i+1)
    plt.scatter(X, y, color='blue', marker='.', alpha=0.5, label='with noise')
    plt.plot(X, true_func(X), color='red', label='true')
    plt.plot(X, y_pred, color='green', label='pred')
    plt.title('h = {}, λ = {}'.format(gauss_width, reg_param))
    plt.text(1.0, 1.0, 'err = {0:.2f}'.format(err))
    if i == 0:
        plt.legend()


def grid_search(X: np.ndarray, y: np.ndarray) -> list:
    """
    最適パラメータのグリッドサーチ
    @param:
        X 説明変数
        y 目的変数
    @return 最適パラメータセット
    """
    param_list = list(itertools.product(GAUSS_WIDTH, REG_PARAM))
    errs = []
    for i, param in enumerate(param_list):
        gauss_width = param[0]
        reg_param = param[1]
        print('gauss_width = {}, reg_param = {}'.format(gauss_width, reg_param))
        err = cross_validation(X, y, gauss_width, reg_param)
        errs.append(err)
        y_pred = predict(X, y, gauss_width, reg_param)
        visualize(X, y, y_pred, gauss_width, reg_param, err, i)
    best_index = np.argmin(errs)
    print('MIN err = {}'.format(np.min(errs)))
    plt.tight_layout()
    plt.show()
    return param_list[best_index]


if __name__ == '__main__':
    main()
