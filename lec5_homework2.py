import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ラベルの種類
NUM_LABEL = 10
# カレントディレクトリのパス
CURRENT_DIR = os.getcwd()
# データが入ったディレクトリのパス
DATA_DIR = os.path.join(os.getcwd(), 'digit/')
# ガウス幅 (ハイパーパラメータ)
H = 1.0
# 正則化パラメータ (ハイパーパラメータ)
L = 1.0


def main() -> None:
    X_train, Y_train = load_data(mode='train')
    X_test, Y_test = load_data(mode='test')
    K_train = gram_matrix(X_train, X_test, mode='train')
    K_test = gram_matrix(X_train, X_test, mode='test')
    Y_pred = []
    for label in range(NUM_LABEL):
        y_train = Y_train[:, label]
        theta = calc_theta(K_train, y_train)
        y_pred = predict(K_test, theta)
        Y_pred.append(y_pred)
    Y_pred = np.array(Y_pred).T
    Y_pred = np.argmax(Y_pred, axis=1).reshape(-1,)
    visualize(Y_test, Y_pred)


def load_data(data_dir: str=DATA_DIR, mode: str='train') -> (np.ndarray, np.ndarray):
    """
    データローディング
    @param:
        data_dir データが入ったディレクトリのパス
        mode train or test
    @return:
        X 説明変数 (256次元のベクトル)
        y 目的変数 (各データに対して、labelの位置が+1、それ以外が-1)
    """
    files = glob.glob(os.path.join(DATA_DIR, 'digit_{}*.csv'.format(mode)))
    files = sorted(files)
    X = []
    Y = []
    for label, f in enumerate(files):
        X_i = pd.read_csv(f, header=None).values.tolist()
        if mode == 'train':
            Y_i = [(2 * np.identity(NUM_LABEL) - np.ones((NUM_LABEL, NUM_LABEL)))[label].tolist()] * len(X_i)
        else:
            Y_i = [label] * len(X_i)
        X += X_i
        Y += Y_i
    return np.array(X), np.array(Y, dtype='int8')


def gauss_kernel(a: np.ndarray, b: np.ndarray, h: float=H) -> float:
    """ ガウスカーネル """
    return np.exp(-np.linalg.norm(a - b) ** 2 / 2 * h ** 2)


def gram_matrix(X_train: np.ndarray, X_test: np.ndarray, h: float=H, mode: str='train') -> np.ndarray:
    """ グラム行列 """
    num_axis = [len(X_test), len(X_train)]
    if mode == 'train':
        num_axis[0] = len(X_train)
    K = np.empty(num_axis)
    for i in range(num_axis[0]):
        for j in range(num_axis[1]):
            if mode == 'train':
                K[i][j] = gauss_kernel(X_train[i], X_train[j], h=h)
            else:
                K[i][j] = gauss_kernel(X_test[i], X_train[j], h=h)
    print('Finish calculating {} gram matrix.'.format(mode))
    return K


def calc_theta(K_train: np.ndarray, y_train: np.ndarray, l: float=L) -> np.ndarray:
    """ 最適重みベクトル """
    A = np.dot(K_train.T, K_train) + l * np.identity(len(K_train.T))
    b = np.dot(K_train.T, y_train)
    return np.linalg.solve(A, b)


def predict(K_test: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """ 予測値 """
    return np.dot(K_test, theta)


def visualize(Y_test: np.ndarray, Y_pred: np.ndarray) -> None:
    """ ヒートマップを可視化 """
    def heatmap(Y_test: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
        count = np.zeros((NUM_LABEL, NUM_LABEL))
        for y_test, y_pred in zip(Y_test, Y_pred):
            count[y_test][y_pred] += 1
        return pd.DataFrame(data=count, index=np.arange(NUM_LABEL), columns=np.arange(NUM_LABEL), dtype='int32')
    hmap = heatmap(Y_test, Y_pred)
    plt.figure()
    sns.heatmap(hmap, vmin=0, annot=True, fmt='d')
    print('A heatmap is showed.')
    plt.show()


if __name__ == '__main__':
    main()
