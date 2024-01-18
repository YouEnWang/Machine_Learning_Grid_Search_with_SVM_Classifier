# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from qpsolvers import solve_qp
import scipy.io
from statistics import mode
from statistics import StatisticsError
# %matplotlib inline

# Separate Versicolor data and Virginica data from Iris data set
def separate(data, change, class_selection):
    # 將資料分為前半(first_half)與後半(second_half)
    split_data = np.split(data, 6)
    first_half = [split_data[0], split_data[2], split_data[4]]
    second_half = [split_data[1], split_data[3], split_data[5]]

    # 根據change更換選擇的training data和test data
    '''select data'''
    if change:                  # 前半資料為training data
        training_data = np.vstack((first_half[class_selection[0] - 1], first_half[class_selection[1] - 1]))
        test_data = np.vstack((second_half[0], second_half[1], second_half[2]))
    else:                       # 後半資料為training data
        training_data = np.vstack((second_half[class_selection[0] - 1], second_half[class_selection[1] - 1]))
        test_data = np.vstack((first_half[0], first_half[1], first_half[2]))

    return training_data, test_data


# 將資料分割為feature與label
def x_y(data):
    # 4個feature全取
    x = data[:, 0:4]
    y = data[:, 4]

    return x, y


# 更改label
def label_change(training_data, class_selection):
    # positive(label=class_selection[0]), negative(label=class_selection[1])
    for i in range(len(training_data)):
        if (training_data[i][4] == class_selection[0]):
            training_data[i][4] = 1
        else:
            training_data[i][4] = -1
    
    return training_data

# Evalute RBF kernel
def RBF_kernel(feature_1, feature_2, sigma):
    x_1 = np.array(feature_1)
    x_2 = np.array(feature_2)
    kernel_set = np.zeros((len(x_1), len(x_2)))
    kernel_set = kernel_set.astype(float)

    r = 1/((2* (sigma**2)))
    
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            # 計算歐幾里德距離
            item = 0.0
            for k in range(len(x_1[i])):
                item = item + np.square(x_1[i][k] - x_2[j][k])
            norm = np.sqrt(item)
            # euclidean_distance = np.linalg.norm(x_1[i] - x_2[j])
            kernel_set[i][j] = np.exp(-r * norm**2)

    return kernel_set


# Evalute Hessian Matrix
def Hessian(label_set, kernel_set):
    y = label_set

    H = np.zeros((len(y), len(y)))
    H = H.astype(float)
    for i in range(len(y)):
        for j in range(len(y)):
            H[i][j] = y[i] * y[j] * kernel_set[i][j]
            
    return H

# Dual problem
def Dual(label_set, Hessian_matrix, C):
    y = label_set
    
    P = Hessian_matrix
    q = -1 * np.ones(len(y))
    A = np.array(y)
    b = np.array([0.0])
    lb = np.zeros(len(y))               # 都是0的一維陣列
    ub = C * np.ones(len(y))            # 都是C的一維陣列

    alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")
    
    eps = 2.2204e-16
    for i in range(alpha.size):
        if alpha[i] >= C - np.sqrt(eps):
            alpha[i] = C
            alpha[i] = np.round(alpha[i],6)
        elif  alpha[i] <= 0 + np.sqrt(eps):
            alpha[i] = 0
            alpha[i] = np.round(alpha[i],6)
        else:
            alpha[i] = np.round(alpha[i],6)
            # print(f"support vector: alpha = {alpha[i]}")
            # print(f"alpha = {np.round(alpha[i],4)}")

    print(alpha[:5])
    return alpha


# Kuhn-Tucker condition
def KT_condition(feature_set, label_set, alpha, C, kernel):
    x = feature_set
    y = label_set
    alpha_set = alpha
    K = kernel

    b_set = np.zeros(len(alpha))
    # print(len(K[0]))
    count = 0
    for i in range(len(alpha_set)):
        if (alpha_set[i] > 0 and alpha_set[i] < C):
            count += 1
            # print(alpha_set[i])
            w_dot_phi = 0
            for j in range(len(y)):
                w_dot_phi += alpha_set[j]* y[j]* K[i][j]
                # print("w_dot_phi = ", w_dot_phi, alpha_set[j], K[i][j])
            b_set[i] = (1/y[i]) - w_dot_phi

    # print("b_set = ", b_set)
    # 計算最佳化的b(平均的b)
    b = 0
    for i in range(len(b_set)):
        b += b_set[i]
    b = round((b/count), 4)

    return b


# 將資料輸出為csv
def output_result(data, filename):
    with open(f"{filename}.csv", "a", newline="") as file:
        for i in range(len(data)):
            file.write(f"{data[i]},")
        file.write("\n")


# Decision Rule
def Decision(y_training, x_test, alpha_set, bias, kernel_set):
    # Prediction result
    prediction = []

    # print(kernel_set)
    
    print("y_training = ", y_training)
    
    for i in range(len(x_test)):
        # Evalute <w, phi>
        w_dot_phi = 0
        D = 0
        for j in range(len(y_training)):
            w_dot_phi += alpha_set[j] * y_training[j] * kernel_set[i][j]
            
        # Decision rule
        D = round(w_dot_phi, 6) + bias
        if (D >= 0):
            prediction.append(1)
        else:
            prediction.append(-1)

    prediction = np.array(prediction)
    
    return prediction


# 計算分類率
def classification_rate(y_test, predict):
    # 預測正確的資料總數
    True_prediction = 0

    # 將predict的label與test data的label做比對
    for i in range(len(predict)):
        if predict[i] == y_test[i]:
            True_prediction += 1
    
    # 分類率
    # print(True_prediction)
    CR = round(True_prediction / len(y_test), 5) * 100
    return CR


# SVM classifier
def SVM(x_training, y_training, kernel, C, sigma):
    '''training process'''
    # Evalute training data的RBF kernel
    training_kernel_set = RBF_kernel(x_training, x_training, sigma)
    # print("training_kernel_set =", training_kernel_set)
    
    # Evalute Hessian Matrix
    H = Hessian(y_training, training_kernel_set)
    print("H =", H)

    # Evalute alpha
    alpha_set = Dual(y_training, H, C)
    num = 0
    for i in range(len(alpha_set)):
        num += alpha_set[i]
        alpha_set[i] = round(alpha_set[i], 4)
    num = round(num, 4)
    # print("num = ", num)
    # print("alpha = ", alpha_set)

    # # 將alpha資料輸出
    # filename = f"alpha_C{C}_sigma{sigma}_SVM"
    
    # output_result(alpha_set, filename)

    # Evalute bias
    b = KT_condition(x_training, y_training, alpha_set, C, training_kernel_set)
    # print("bias = ", b)

    # 將結果繪圖
    # scatter_plot(training_data, test_data, alpha_set, b, C, sigma)

    return alpha_set, b


# 多類別分類
def SVM_mult(initial_data, change, kernel, C, sigma):
    # 儲存預測結果
    predict_set = []

    class_set = [[1, 2], [1, 3], [2, 3]]
    for i in range(len(class_set)):
        '''training process'''
        # Split data
        training_data, test_data = separate(initial_data, change, class_set[i])

        # 更改label為1, -1
        training_data = label_change(training_data, class_set[i])
        
        # 將資料的feature跟label分離
        x_training, y_training = x_y(training_data)
        x_test, y_test = x_y(test_data)
        
        # 執行SVM
        alpha_set, b = SVM(x_training, y_training, kernel, C, sigma)
        
        '''testing process'''
        # Evalute test data的kernel
        test_kernel_set = RBF_kernel(x_test, x_training, sigma)
        
        # 預測結果
        prediction = Decision(y_training, x_test, alpha_set, b, test_kernel_set)
        print("prediction = ", prediction)
        # print(len(prediction))
        
        # 將label(1, -1)換回1, 2,or 3
        for j in range(len(prediction)):
            if prediction[j] == 1:
                prediction[j] = class_set[i][0]
            else:
                prediction[j] = class_set[i][1]
        
        # # 將prediction輸出
        # filename = f"C{C}_sigma{sigma}_class"
        # output_result(prediction, filename)

        predict_set.append(prediction)
    
    predict_set = np.array(predict_set)
    print("predict_set", predict_set.T)

    # 多數決後的預測結果
    predict_final = []

    for i in range(len(predict_set.T)):
        try:
            find_mode = mode(predict_set.T[i])      # Voting
            predict_final.append(find_mode)
        except StatisticsError:
            # 如果 mode() 引發错误，表示Voting結果有產生平手
            predict_final.append("Error")   # 判定為分類錯誤
            continue
    
    # 計算分類率
    CR = classification_rate(y_test, predict_final)

    return predict_final, CR


# 繪製結果
def scatter_plot(training_data, test_data, alpha_set, b, C, sigma):
    # 提取特徵和標籤
    x_training, y_training = x_y(training_data)
    x_test, y_test = x_y(test_data)

    # 計算支持向量
    support_vectors = []
    for i, alpha in enumerate(alpha_set):
        if alpha > 0:
            support_vectors.append(x_training[i])

    support_vectors = np.array(support_vectors)

    # 繪製散點圖
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.size'] = 14
    plt.title(f'SVM Classifier (C={C}, sigma={sigma})')

    plt.scatter(x_training[y_training == 1][:, 0], x_training[y_training == 1][:, 1], c='b', marker='o', label='Positive_training')
    plt.scatter(x_training[y_training == -1][:, 0], x_training[y_training == -1][:, 1], c='b', marker='x', label='Negative_training')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='y', marker='*', s=100, label='Support Vectors')

    # 創立網格來覆蓋測試數據的空間
    x_min, x_max = np.amin(x_test.T[0]) - 1, np.amax(x_test.T[0]) + 1
    y_min, y_max = np.amin(x_test.T[1]) - 1, np.amax(x_test.T[1]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # 對網格上的點進行預測
    Z = []
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    kernel_set = RBF_kernel(grid_points, x_training, sigma)
    Z = [Decision(training_data, grid_points, alpha_set, b, C, kernel_set)]
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)

    # 畫決策邊界和測試樣本點
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, levels=[-1, 0, 1], alpha=0.8)
    plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], c='r', marker='o', label='Positive_test')
    plt.scatter(x_test[y_test == -1][:, 0], x_test[y_test == -1][:, 1], c='r', marker='x', label='Negative_test')
    
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(scatterpoints=1, markerscale=1, loc='lower right')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# main function
def main():
    # load data
    raw_data = np.loadtxt("iris.txt", dtype=float)

    # 參數設定
    kernel = "RBF"
    C_list = [1, 5, 10, 50, 100, 500, 1000]
    change = [0, 1]         # 做2-fold CV
    
    '''做grid search !!!'''
    # 更動C
    for C in C_list:
        power = -100                    # sigma的次方數
        CR_set = [0.0, 0.0]             # 儲存2-fold CV的 CR

        # 將資料輸出   
        filename = f"prediction_RBF_C{C}_SVM"
        title = ["C", "sigma", "CR"]
        with open(f"{filename}.csv", "a", newline="") as file:
            for i in range(len(title)):
                file.write(f"{title[i]}, ")
            file.write("\n")

            # 更動sigma
            for i in range(41):
                sigma = math.pow(1.05, power)

                # 2-fold cross validation
                for j in change:
                    prediction, CR = SVM_mult(raw_data, j, kernel, C, sigma)
                    CR_set[j] = CR
                    print(f"prediction_{j+1} = ", prediction)
                    print(f"CR_{j+1} = ", CR)

                # 計算blanced CR
                CR_balanced = (CR_set[0] + CR_set[1]) / 2

                # 將資料輸出
                file.write(f"{C}, {sigma}, {'%.2f'%round(CR_balanced, 2)} %\n")
                
                power = power + 5

        file.close()

    
if __name__ == "__main__":
    main()