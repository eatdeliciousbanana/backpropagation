# backpropagations.py
# バックプロパゲーションによるニューラルネットの学習
# 誤差の推移や，学習結果となる結合係数などを出力します


# モジュール
import random  # 乱数
import pathlib  # ファイル操作
import math  # 数学関数


# 定数
INPUTNO = 10  # 入力層のセル数
HIDDENNO = 10  # 中間層のセル数
OUTNO = 2  # 出力層のセル数
MU = 5.0  # 学習係数
BIGNUM = 100  # 誤差の初期値
LIMIT = 0.001  # 誤差の上限値
TEACH_FILENAME = "teaching_data.txt"  # 入力データファイル
TEST_FILENAME = "unknown_data.txt"  # 未知データファイル


# main関数
def main():
    wh = []  # 中間層の重み
    wo = []  # 出力層の重み
    e = []  # 学習データセット
    test = []  # 未知データセット
    n_of_e = 0  # 学習データの個数
    n_of_test = 0  # 未知データの個数
    zh = [0]*HIDDENNO  # 中間層の出力
    zo = [0]*OUTNO  # 出力
    err = BIGNUM  # 誤差の評価
    count = 0  # 繰り返し回数のカウンタ

    # 重みの初期化
    initwh(wh)
    initwo(wo)

    # 学習データの読み込み
    n_of_e = get_edata(e)

    # 学習
    while err > LIMIT:
        err = 0.0
        for j in range(n_of_e):
            # 順方向の計算
            forward(zo, wh, wo, zh, e[j])
            # 出力層の重みの調整
            olearn(wo, zh, e[j], zo)
            # 中間層の重みの調整
            hlearn(wh, wo, zh, e[j], zo)
            # 誤差の積算
            for i in range(OUTNO):
                err += (zo[i]-e[j][INPUTNO+i])*(zo[i]-e[j][INPUTNO+i])
        count += 1
        # 誤差の出力
        print("{:<8d}{:f}".format(count, err))
    # 学習終了

    # 学習データに対する出力
    teach_result(wh, wo, zh, zo, e, n_of_e)

    # 未知データの読み込み
    n_of_test = get_testdata(test)

    # 未知データに対する出力
    test_result(wh, wo, zh, zo, test, n_of_test)


# 中間層の重みの初期化
def initwh(wh):
    for i in range(HIDDENNO):
        temp = []
        for j in range(INPUTNO + 1):
            temp.append(drnd())
        wh.append(temp)


# 出力層の重みの初期化
def initwo(wo):
    for i in range(OUTNO):
        temp = []
        for j in range(HIDDENNO+1):
            temp.append(drnd())
        wo.append(temp)


# 乱数の生成
def drnd():
    # -1から1の間の乱数を生成
    return random.uniform(-1.0, 1.0)


# 入力データの読み込み
def get_edata(e):
    p = pathlib.Path("./"+TEACH_FILENAME)
    f = p.open("r")
    n_of_e = 0  # データセットの個数
    while True:
        data = f.readline()
        if data == "":
            break
        e.append(data.rstrip("\n").split())
        n_of_e += 1
    f.close()
    for i in range(n_of_e):
        for j in range(INPUTNO+OUTNO):
            e[i][j] = float(e[i][j])
    return n_of_e


# 順方向の計算
def forward(zo, wh, wo, zh, e):
    u = 0  # 重み付き和の計算

    # zhの計算
    for i in range(HIDDENNO):
        u = 0
        # 伝達関数へ渡す値uを計算する処理
        for j in range(INPUTNO):
            u += wh[i][j]*e[j]
        u -= wh[i][INPUTNO]
        zh[i] = f(u)

    # 出力zoの計算
    for i in range(OUTNO):
        u = 0
        # 伝達関数へ渡す値uを計算する処理
        for j in range(HIDDENNO):
            u += wo[i][j]*zh[j]
        u -= wo[i][HIDDENNO]
        zo[i] = f(u)


# 伝達関数（シグモイド関数）
def f(u):
    return 1.0/(1.0+math.exp(-u))


# 出力層の重み学習
def olearn(wo, zh, e, zo):
    di = 0  # 重み計算に利用
    for i in range(OUTNO):
        di = (zo[i]-e[INPUTNO+i])*zo[i]*(1-zo[i])  # 誤差の計算
        for j in range(HIDDENNO):
            wo[i][j] += (-1.0)*MU*di*zh[j]  # 重みの学習
        wo[i][HIDDENNO] += MU*di  # 閾値の学習


# 中間層の重み学習
def hlearn(wh, wo, zh, e, zo):
    dik = 0  # 中間層の重み計算に利用
    di = 0  # 中間層の重み計算に利用

    # 中間層の各セルiを対象
    for i in range(HIDDENNO):
        dik = 0
        for k in range(OUTNO):
            dik += (zo[k]-e[INPUTNO+k])*zo[k]*(1-zo[k])*wo[k][i]

        di = dik*zh[i]*(1-zh[i])  # 誤差の計算

        # 重みと閾値wh[i][j]を更新する処理
        for j in range(INPUTNO):
            wh[i][j] += (-1.0)*MU*di*e[j]  # 重みの学習
        wh[i][INPUTNO] += MU*di  # 閾値の学習


# 学習データに対する出力
def teach_result(wh, wo, zh, zo, e, n_of_e):
    print("学習データの個数:"+str(n_of_e))
    print("--teaching_data--")
    for i in range(n_of_e):
        print("{:<5d}".format(i), end="")
        for j in range(INPUTNO):
            print("{:f} ".format(e[i][j]), end="")
        forward(zo, wh, wo, zh, e[i])
        print("| ", end="")
        for j in range(OUTNO):
            print("{:f} ".format(zo[j]), end="")
        print()


# 未知データの読み込み
def get_testdata(test):
    p = pathlib.Path("./"+TEST_FILENAME)
    f = p.open("r")
    n_of_test = 0  # データセットの個数
    while True:
        data = f.readline()
        if data == "":
            break
        test.append(data.rstrip("\n").split())
        n_of_test += 1
    f.close()
    for i in range(n_of_test):
        for j in range(INPUTNO):
            test[i][j] = float(test[i][j])
    return n_of_test


# 未知データに対する出力
def test_result(wh, wo, zh, zo, test, n_of_test):
    print("未知データの個数:"+str(n_of_test))
    print("--unknown_data--")
    for i in range(n_of_test):
        print("{:<5d}".format(i), end="")
        for j in range(INPUTNO):
            print("{:f} ".format(test[i][j]), end="")
        forward(zo, wh, wo, zh, test[i])
        print("| ", end="")
        for j in range(OUTNO):
            print("{:f} ".format(zo[j]), end="")
        print()


# main関数の呼び出し
if __name__ == "__main__":
    main()
