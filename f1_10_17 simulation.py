# -*- coding: utf-8 -*-
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from itertools import product
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import time

start_time = time.time()
### 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 1. 生成地理位置 s (假设每个样本有2个地理特征，如经度和纬度)
n_samples = 2000
s = np.random.uniform(low=-10, high=10, size=(n_samples, 2))  # 随机生成经纬度位置

# 2. 生成特征矩阵 X (2000 x 5), 随机生成符合正态分布的特征数据
n_features = 5
X = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))

# 3. 随机生成 beta 向量 (5维, 与X的列数对应)
beta = np.random.normal(loc=0, scale=1, size=(n_features, 1))

# 4. 定义 Matérn 协方差函数
def matern_covariance(d, theta):
    length_scale, amplitude, smoothness = theta
    if smoothness == 0.5:
        return amplitude * np.exp(-d / length_scale)
    elif smoothness == 1.5:
        sqrt_3_d_theta = np.sqrt(3) * d / length_scale
        return amplitude * (1 + sqrt_3_d_theta) * np.exp(-sqrt_3_d_theta)
    elif smoothness == 2.5:
        sqrt_5_d_theta = np.sqrt(5) * d / length_scale
        return amplitude * (1 + sqrt_5_d_theta + (5 / 3) * (sqrt_5_d_theta ** 2)) * np.exp(-sqrt_5_d_theta)
    else:
        raise ValueError("Smoothness value must be 0.5, 1.5, or 2.5.")

# 5. 定义协方差矩阵生成函数
def covariance_matrix(theta, s):
    dist_matrix = squareform(pdist(s, metric='euclidean'))  # 计算距离矩阵
    return matern_covariance(dist_matrix, theta)

# 6. 定义参数 θ = (length_scale, amplitude, smoothness)
theta = (1.0, 1.0, 1.5)

# 7. 生成协方差矩阵 Σ(θ)
Sigma_theta = covariance_matrix(theta, s)

# 8. 计算均值向量 μ = Xβ
mu = X @ beta

# 9. 使用 Cholesky 分解生成 y ~ N(Xβ, Σ(θ))
L = cholesky(Sigma_theta + 1e-6 * np.eye(n_samples), lower=True)  # 增加小扰动确保矩阵正定
y = mu + L @ np.random.normal(size=(n_samples, 1))

# 打印结果
print(f"Generated y with shape: {y.shape}")


# 生成基于经纬度的协方差矩阵
def matern_covariance(d, theta):
    length_scale, amplitude, nu = theta
    if nu == 0.5:
        cov = amplitude * np.exp(-d / length_scale)
    elif nu == 1.5:
        sqrt_3_d_theta = np.sqrt(3) * d / length_scale
        cov = amplitude * (1 + sqrt_3_d_theta) * np.exp(-sqrt_3_d_theta)
    elif nu == 2.5:
        sqrt_5_d_theta = np.sqrt(5) * d / length_scale
        cov = amplitude * (1 + sqrt_5_d_theta + (5 / 3) * (sqrt_5_d_theta ** 2)) * np.exp(-sqrt_5_d_theta)
    else:
        raise ValueError("Unsupported nu value. Use 0.5, 1.5, or 2.5.")
    return cov

def covariance_matrix(theta, longitudes, latitudes):
    positions = np.column_stack([longitudes, latitudes])
    dist_matrix = squareform(pdist(positions, 'euclidean'))
    cov_matrix = matern_covariance(dist_matrix, theta)
    return cov_matrix

# 第一阶段：使用线性回归 + GLS 估计均值函数 m(x)
def GLS_regression(theta, X, s, y):
    cov_matrix_train = covariance_matrix(theta, s[:, 0], s[:, 1])
    
    # 定义 jitter 值，确保矩阵是正定的
    jitter = 1e-4
    chol_factor = cholesky(cov_matrix_train + jitter * np.eye(cov_matrix_train.shape[0]), lower=True)

    # 使用Cholesky分解对特征进行预处理
    X_gls = np.linalg.solve(chol_factor, np.hstack((X, s)))
    y_gls = np.linalg.solve(chol_factor, y)

    # 线性回归模型拟合
    regression_model = LinearRegression()
    regression_model.fit(X_gls, y_gls)
    return regression_model, chol_factor, jitter

# 第二阶段：定义DNN模型
def create_dnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))  # 使用Input层来指定输入形状
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))  # 输出为1个预测值
    return model

# 定义广义最小二乘损失函数，使用TensorFlow操作
def GLS_loss(y_true, y_pred, chol_factor):
    residual = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
    # 将 chol_factor 转换为 float32 类型
    chol_factor = tf.cast(chol_factor, tf.float32)
    # 调整批次中的残差尺寸以匹配 chol_factor
    batch_size = tf.shape(residual)[0]
    chol_factor_batch = chol_factor[:batch_size, :batch_size]
    
    # 使用 TensorFlow 的矩阵求解操作
    residual_adjusted = tf.linalg.triangular_solve(chol_factor_batch, residual, lower=True)
    return tf.reduce_mean(tf.square(residual_adjusted))

# 网格搜索的参数
length_scales = np.linspace(0.1, 100, 100)
amplitudes = np.linspace(0.1, 10, 15)
smoothness_values = [0.5, 1.5]
parameter_grid = list(product(length_scales, amplitudes, smoothness_values))

# 记录最佳模型和参数
best_loss = float('inf')
best_model = None
best_theta = None

iteration_count = 1  # 初始化计数器

# 保存结果的文件路径
results_file = "parameter_search_results.csv"

# 创建一个空的 DataFrame 来保存参数和损失
results_df = pd.DataFrame(columns=['Iteration', 'Theta_length_scale', 'Theta_amplitude', 'Theta_smoothness', 'val_loss'])

for theta in parameter_grid:
    
    print(f"Iteration: {iteration_count}, Current Theta: {theta}")
    
    # 第一阶段：线性回归 + GLS
    regression_model, chol_factor_train, jitter = GLS_regression(theta, X_train, s_train, y_train)
    
    # 计算线性回归的初步预测
    X_train_gls = np.linalg.solve(chol_factor_train, np.hstack((X_train, s_train)))
    y_pred_train = regression_model.predict(X_train_gls)

    # 计算残差
    residuals_train = y_train - y_pred_train

    # 标准化数据（特征 + 空间位置）
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(np.hstack((X_train, s_train)))

    # 将残差加入到输入
    X_train_with_residuals = np.hstack((X_train_normalized, residuals_train))

    # 第二阶段：通过 DNN 进一步优化
    dnn_model = create_dnn_model(X_train_with_residuals.shape[1])  # 输入维度为原始维度 + 残差
    dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss=lambda y_true, y_pred: GLS_loss(y_true, y_pred, chol_factor_train))

    # 训练 DNN
    dnn_model.fit(X_train_with_residuals, y_train, epochs=10, batch_size=32, verbose=0)

    # 计算验证集的损失
    chol_factor_val = cholesky(covariance_matrix(theta, s_val[:, 0], s_val[:, 1]) + jitter * np.eye(len(s_val)), lower=True)

    
    # 计算线性回归的初步预测（验证集上的预测）
    X_val_gls = np.linalg.solve(chol_factor_val, np.hstack((X_val, s_val)))
    y_pred_val_regression = regression_model.predict(X_val_gls)

  # 计算验证集的残差
    residuals_val = y_val - y_pred_val_regression

# 标准化验证集数据
    X_val_normalized = scaler.transform(np.hstack((X_val, s_val)))

# 将残差加入到验证集输入
    X_val_with_residuals = np.hstack((X_val_normalized, residuals_val))

# 使用 DNN 模型进行预测
    y_pred_val = dnn_model.predict(X_val_with_residuals)
    val_loss = GLS_loss(y_val, y_pred_val, chol_factor_val)
    print(f"New val Theta: {theta}, val_Loss: {val_loss}") 
    
# 使用 pandas.concat 代替 append
    new_row = pd.DataFrame([{
       'Iteration': iteration_count,
       'Theta_length_scale': theta[0],
       'Theta_amplitude': theta[1],
       'Theta_smoothness': theta[2],
      'val_loss': val_loss.numpy()
    }])

    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # 将结果写入文件，每次迭代后都保存
    results_df.to_csv(results_file, index=False)
    iteration_count += 1
    # 更新最优模型和参数
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = dnn_model
        best_theta = theta
      # 确认每次执行后完成
        # 增加调试信息
        print(f"New best Theta: {best_theta}, Loss: {best_loss}")  # 增加调试信息

if best_theta is not None:
    print(f"Best Theta: {best_theta}, Best Loss: {best_loss}")
else:
    print("No valid Theta found during the search.")


end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time} seconds")

#Cholesky decomposition failed. Matrix might not be positive definite.
#Best Negative Log-Likelihood: -inf
#Best estimated theta: [0.08120428 0.10725828 2.5336003 ]
#Best MSE: 0.023111627125256877
#回归系数 (Regression Coefficients): [[ 0.18292052 -2.62354667  2.39537107 -1.54152053  1.30191826  1.41135568]]
#截距 (Intercept): [-0.05135543]
#Total runtime: 39489.94802355766 seconds 11h  40*30*3
###MSE   回归系数 都没变就是最开始得到的

#Iteration: 50, Current Theta: (1.0, 1.0, 1.5)
#New val Theta: (1.0, 1.0, 1.5), val_Loss: 202.44125366210938
#Best Theta: (0.1, 0.775, 0.5), Best Loss: 1.8128281831741333
#Total runtime: 109.65474796295166 seconds

#Iteration: 140, Current Theta: (10.0, 10.0, 1.5)
#New val Theta: (10.0, 10.0, 1.5), val_Loss: 269.7847595214844
#Best Theta: (0.1, 5.6, 0.5), Best Loss: 0.528283953666687
#Total runtime: 321.4512176513672 seconds

#New val Theta: (100.0, 10.0, 1.5), val_Loss: 333.2684020996094
#Best Theta: (0.1, 10.0, 0.5), Best Loss: 0.9758087396621704

#New val Theta: (100.0, 10.0, 0.5), val_Loss: 24.655807495117188
#Iteration: 3000, Current Theta: (100.0, 10.0, 1.5)
#New val Theta: (100.0, 10.0, 1.5), val_Loss: 309.396728515625
#Best Theta: (0.1, 10.0, 0.5), Best Loss: 1.0391384363174438
#Total runtime: 8109.675303697586 seconds 2h
