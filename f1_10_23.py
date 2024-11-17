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
from tensorflow.keras.layers import Dense, Dropout, Input
from itertools import product
from tensorflow.keras.optimizers import Adam
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

start_time = time.time()

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 加载并预处理实际数据
df = pd.read_csv('C:/Users/24304/Downloads/CaliforniaHousing/cal_housing.data', delimiter=',')
new_column_names = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                    'totalBedrooms', 'population', 'households', 'medianIncome', 'medianHouseValue']
df.columns = new_column_names

# 对某些列进行 log 变换，减少数据的偏度
for col in ['totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome']:
    df[col] = np.log(df[col] + 1)

# 特征和目标分离
X = df[['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome']].values
s = df[['longitude', 'latitude']].values  # 地理位置（经度和纬度）
y = df['medianHouseValue'].values.reshape(-1, 1)  # 目标变量

# 归一化特征
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

scaler_s = MinMaxScaler()
s = scaler_s.fit_transform(s)

scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)

# 手动指定训练集、验证集和测试集的大小
test_size = 200
validation_size = 18000

# 划分训练集和测试集
X_train_val, X_test, s_train_val, s_test, y_train_val, y_test = train_test_split(
    X, s, y, test_size=test_size, random_state=42)

# 再次划分训练集和验证集
X_train, X_val, s_train, s_val, y_train, y_val = train_test_split(
    X_train_val, s_train_val, y_train_val, test_size=validation_size, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

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

# 优化后的 covariance_matrix 函数，确保 float32
def optimized_covariance_matrix(theta, longitudes, latitudes):
    cov_matrix = covariance_matrix(theta, longitudes, latitudes).astype(np.float32)
    return cov_matrix

# 第一阶段：使用线性回归 + GLS 估计均值函数 m(x)
def GLS_regression(theta, X, s, y):
    cov_matrix_train = optimized_covariance_matrix(theta, s[:, 0], s[:, 1])
    
    # 定义 jitter 值，确保矩阵是正定的
    jitter = 1e-2
    chol_factor = cholesky(cov_matrix_train + jitter * np.eye(cov_matrix_train.shape[0], dtype=np.float32), lower=True)
    
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

# 使用标准的均方误差损失函数
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 自定义 GLS_loss 函数，使用 reduce_mean 代替 reduce_sum
def GLS_loss(y_true, y_pred, chol_factor):
    # 确保 chol_factor 是 TensorFlow 常量，避免每次调用时转换
    chol_factor_tf = tf.constant(chol_factor, dtype=tf.float32)
    
    # 计算残差
    residual = y_true - y_pred  # Shape: (batch_size, 1)
    
    # 使用 triangular_solve 计算调整后的残差
    residual_adjusted = tf.linalg.triangular_solve(chol_factor_tf, residual, lower=True)
    
    # 计算均方误差
    loss = tf.reduce_mean(tf.square(residual_adjusted))
    return loss

# 网格搜索的参数
length_scales = np.linspace(0.1, 100, 50)
amplitudes = np.linspace(0.1, 100, 50)
smoothness_testues = [0.5]
parameter_grid = list(product(length_scales, amplitudes, smoothness_testues))

# 记录最佳模型和参数
best_loss = float('inf')
best_model = None
best_theta = None

iteration_count = 1  # 初始化计数器

# 保存结果的文件路径
results_file = "parameter_search_results.csv"

# 创建一个空的 DataFrame 来保存参数和损失
results_df = pd.DataFrame(columns=['Iteration', 'Theta_length_scale', 'Theta_amplitude', 'Theta_smoothness', 'test_loss'])

for theta in parameter_grid:
    print(f"Iteration: {iteration_count}, Current Theta: {theta}")
    
    # 第一阶段：线性回归 + GLS
    regression_model, chol_factor_train, jitter = GLS_regression(theta, X_train, s_train, y_train)
    
    # 计算线性回归的初步预测
    X_train_gls = np.linalg.solve(chol_factor_train, np.hstack((X_train, s_train)))
    y_pred_train = regression_model.predict(X_train_gls)
    
    # 计算残差
    residuals_train = y_train - y_pred_train
    
    # 标准化数据（特征 + 空间位置）  第二阶段 
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(np.hstack((X_train, s_train)))
    
    # 所以 DNN 的输入为标准化后的特征，目标为 residuals_train
    X_train_with_features = X_train_normalized  # DNN 输入
    y_train_residuals = residuals_train  # DNN 目标
    
    # 第二阶段：通过 DNN 进一步优化
    dnn_model = create_dnn_model(X_train_with_features.shape[1])  # 输入维度为标准化后的特征
    dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # 使用标准的 MSE 损失
    
    # 增加训练轮数并监控训练过程
    history = dnn_model.fit(
        X_train_with_features, y_train_residuals,
        epochs=50,  # 增加 epochs 数量
        batch_size=len(X_train_with_features),  # 全批量训练
        verbose=0  # 如果需要观察训练过程，可以设置为 1
    )
    
    #=dnn_model.predict(X_train_with_features)
    # 计算测试集的损失
    # GLS 预测
    cov_matrix_test = optimized_covariance_matrix(theta, s_test[:, 0], s_test[:, 1])
    chol_factor_test = cholesky(cov_matrix_test + jitter * np.eye(len(s_test), dtype=np.float32), lower=True)
    
    X_test_gls = np.linalg.solve(chol_factor_test, np.hstack((X_test, s_test)))
    y_pred_test_regression = regression_model.predict(X_test_gls)
    
    # 计算 GLS 残差
    residuals_test = y_test - y_pred_test_regression
    
    # 标准化测试集数据
    X_test_normalized = scaler.transform(np.hstack((X_test, s_test)))
    
    # 使用 DNN 模型预测残差
    y_pred_test_residuals = dnn_model.predict(X_test_normalized)
    
    # 最终预测
    y_pred_test_final = y_pred_test_regression + y_pred_test_residuals
    
    # 计算最终的 MSE 损失
    test_loss = mse_loss(y_test, y_pred_test_final)
    print(f"New test Theta: {theta}, test_Loss: {test_loss.numpy()}") 
    
  
    
    new_row = pd.DataFrame([{
       'Iteration': iteration_count,
       'Theta_length_scale': theta[0],
       'Theta_amplitude': theta[1],
       'Theta_smoothness': theta[2],
       'test_loss': test_loss.numpy()
    }])

    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # 将结果写入文件
    results_df.to_csv(results_file, index=False)
    iteration_count += 1
    
    # 更新最优模型和参数
    if test_loss < best_loss:
        best_loss = test_loss
        best_model = dnn_model
        best_theta = theta
        print(f"New best Theta: {best_theta}, Loss: {best_loss}")  # 增加调试信息

# 最佳参数后续处理
if best_theta is not None:
    print(f"Best Theta: {best_theta}, Best Loss: {best_loss}")
    
    # 逆归一化预测值和真实值
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_test_final_inv = scaler_y.inverse_transform(y_pred_test_final.reshape(-1, 1)).flatten()
    
    # 确保数据是一维的
    y_test_inv = y_test_inv.flatten()
    y_pred_test_final_inv = y_pred_test_final_inv.flatten()

    # 绘制真实值与预测值的散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_inv, y_pred_test_final_inv, alpha=0.5, label='Predictions')
    plt.xlabel('True Median House Value')
    plt.ylabel('Predicted Median House Value')
    plt.title('True vs Predicted Median House Values on Test Set')
    plt.legend()

    # 绘制 45 度参考线
    min_val = min(y_test_inv.min(), y_pred_test_final_inv.min())
    max_val = max(y_test_inv.max(), y_pred_test_final_inv.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')
    plt.legend()

    plt.show()
    
    # 计算训练集前500个样本的损失
    X_train_subset = X_train[:500]
    s_train_subset = s_train[:500]
    y_train_subset = y_train[:500]
    
    # 计算协方差矩阵和 Cholesky 分解
    cov_matrix_train_subset = optimized_covariance_matrix(best_theta, s_train_subset[:, 0], s_train_subset[:, 1])
    chol_factor_train_subset = cholesky(cov_matrix_train_subset + jitter * np.eye(len(s_train_subset), dtype=np.float32), lower=True)
    
    # 计算线性回归的初步预测
    X_train_subset_gls = np.linalg.solve(chol_factor_train_subset, np.hstack((X_train_subset, s_train_subset)))
    y_pred_train_subset_regression = regression_model.predict(X_train_subset_gls)
    
    # 计算 GLS 残差
    residuals_train_subset = y_train_subset - y_pred_train_subset_regression
    
    # 标准化数据
    X_train_subset_normalized = scaler.transform(np.hstack((X_train_subset, s_train_subset)))
    
    # 使用最佳 DNN 模型预测残差
    y_pred_train_subset_residuals = best_model.predict(X_train_subset_normalized)
    
    # 最终预测
    y_pred_train_subset_final = y_pred_train_subset_regression + y_pred_train_subset_residuals
    
    # 计算最终的 MSE 损失
    train_subset_loss = mse_loss(y_train_subset, y_pred_train_subset_final)
    print(f"Loss on first 500 training samples with best Theta: {train_subset_loss.numpy()}")
else:
    print("No valid Theta found during the search.")

mean_predicted_y_test =np.mean(y_pred_test_final)
print(f"Best predicted mean function on test set:{mean_predicted_y_test}")
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time} seconds")

#Best Theta: [0.1, 0.1, 0.5], Best Loss: 0.28460019146767473
#Loss on first 500 training samples with best Theta: 0.4930477016180848
#Best predicted mean function on test set:0.47773314430663416
#Total runtime: 239.42555570602417 seconds

#Best Theta: (0.1, 67.37959183673469, 0.5), Best Loss: 0.02308466279256508
#Loss on first 500 training samples with best Theta: 0.026564682287619143
#Best predicted mean function on test set:0.3705372223568512
#Total runtime: 7331.850136995316 seconds