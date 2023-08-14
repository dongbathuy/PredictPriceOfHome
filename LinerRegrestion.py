import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('housing1.csv')
x = dataframe.values[:, 5]
y = dataframe.values[:, 13]
plt.scatter(x, y, marker='o')
plt.xlabel("Số phòng trung bình")
plt.ylabel("Giá nhà")
plt.show()
# Xác định hàm loại bỏ điểm nhiễu bằng Z-score
def remove_outliers_zscore(x, y, threshold):
    z_scores_x = np.abs((x - np.mean(x)) / np.std(x))
    z_scores_y = np.abs((y - np.mean(y)) / np.std(y))
    filtered_indices = np.where((z_scores_x <= threshold) & (z_scores_y <= threshold))
    x_filtered, y_filtered = x[filtered_indices], y[filtered_indices]
    return x_filtered, y_filtered

# Loại bỏ các điểm nhiễu từ dữ liệu gốc
threshold_zscore = 2.5  # Điều chỉnh ngưỡng Z-score theo yêu cầu của bạn
x_filtered, y_filtered = remove_outliers_zscore(x, y, threshold_zscore)

x_train, x_test, y_train, y_test = train_test_split(x_filtered, y_filtered, test_size=0.3, random_state=42)
plt.scatter(x_train, y_train, marker='o')
plt.xlabel("Số phòng trung bình")
plt.ylabel("Giá nhà")
plt.show()
plt.scatter(x_test, y_test, marker='o')
plt.xlabel("Số phòng trung bình")
plt.ylabel("Giá nhà")
plt.show()
def predict(new_radio, weight, bias):
    return weight * new_radio + bias

# Ham chi phi
def cost_function(x, y, weight, bias):  # MSE
    n = len(x)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - weight * x[i] - bias) ** 2
    return sum_error / n

# Ung dung cua gradient descent de cap nhat weight
def update_weight(x, y, weight, bias, learning_rate):
    n = len(x)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2 * x[i] * (y[i] - (x[i] * weight + bias))
        bias_temp += -2 * (y[i] - (x[i] * weight + bias))
    weight -= (weight_temp / n) * learning_rate
    bias -= (bias_temp / n) * learning_rate
    return weight, bias

# Ham huan luyen
def train(x, y, weight, bias, learning_rate, iter):
    cos_his = []
    alpha = 0.000000005
    beta = 0.00005  
    learning_rate_new= learning_rate
    for i in range(iter):
        cost = cost_function(x,y,weight,bias)
        weight_old= weight
        bias_old = bias
        weight, bias = update_weight(x, y, weight, bias, learning_rate)
        cost_new = cost_function(x, y, weight, bias)

        while cost_new - cost > - alpha * learning_rate_new * (np.linalg.norm(weight_old)**2 + np.linalg.norm(bias_old)**2): 
            learning_rate_new= learning_rate*beta
            weight, bias = update_weight(x, y, weight, bias, learning_rate_new)
            cost_new = cost_function(x,y,weight,bias)
            print(weight, bias)
            print("cost", cost_new)
            print(1)
        if math.sqrt(np.linalg.norm(weight_old)**2 + np.linalg.norm(bias_old)**2)<0.00000000008:
            print(0)
            break
        print ("Kết thúc ", i)
        cos_his.append(cost_new)
        learning_rate =learning_rate_new
    return weight, bias, cos_his

# Huan luyen mo hinh
weight, bias, cost = train(x_train, y_train, 11.463789638460343,
-50.207084349328646, 1, 3000)

print('Kết quả:')
print(weight)
print(bias)
print(cost)


solanlap = [i for i in range(len(cost))]
plt.plot(solanlap, cost)
plt.show()

y_pred = predict(x_test, weight, bias)
# Ve duong du doan
plt.scatter(x_test, y_test, marker='o', label='Dữ liệu kiểm tra')
plt.xlabel("Số phòng trung bình")
plt.ylabel("Giá nhà")
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Đường dự đoán')
plt.legend()
plt.show()
print('Giá trị dự đoán:')
print(predict(5.631, weight, bias))

 # Danh gia mo hinh
def calculate_r_squared(y, y_pred):
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

# Tinh toan he so R - squared
r_squared = calculate_r_squared(y_test, y_pred)
print('Hệ số R-squared:', r_squared)
