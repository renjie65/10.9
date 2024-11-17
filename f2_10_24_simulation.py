# -*- coding: utf-8 -*-
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from itertools import product
from scipy.special import gamma, kv
from scipy.spatial.distance import cdist
import time

start_time = time.time()

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 如果需要抽取多个样本，可以指定 size 参数
D=100
n=3000
s= np.random.uniform(0, D, size=n)
x_1=s/D
x_2=np.sin(10*s/D)
x_3=(s/D)**2
x_4=np.exp(3*s/D)
x_5=1/(s/D+1)

s=s.reshape(n,1)
x_1=x_1.reshape(n,1)
x_2=x_2.reshape(n,1)
x_3=x_3.reshape(n,1)
x_4=x_4.reshape(n,1)
x_5=x_5.reshape(n,1)
x=np.hstack([x_1,x_2,x_3,x_4,x_5])#3000*5

def matern_covariance(s, theta):
    length_scale, sigma,nu=theta
    dist_matrix = cdist(s, s)
    r=dist_matrix
    factor = (2**(1-nu)) / gamma(nu)
    scale_term = (np.sqrt(2 * nu) * r + 1e-10) / length_scale
    kv(nu, scale_term)
    return sigma**2 * factor * (scale_term**nu) * kv(nu, scale_term)

ture_theta=[10.0,10.0,0.5]
dis_matrix = cdist(s, s)
cov = matern_covariance(dis_matrix, ture_theta)
####无法计算协方差阵
mean = np.zeros(n) 
e1 = np.random.multivariate_normal(mean, cov).reshape(n,1)

##### 生成数据y
f0= np.sum(x, axis=1)
f0=f0.reshape(n,1)
y=f0+e1

# 归一化特征和目标变量
scaler_X = MinMaxScaler()
x = scaler_X.fit_transform(x)

scaler_s = MinMaxScaler()
s = scaler_s.fit_transform(s)

scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)

# 手动指定测试集的大小
test_size = 500

# 划分训练集和测试集
x_train, x_test, s_train, s_test, y_train, y_test = train_test_split(x, s, y, test_size=test_size, random_state=42)

print(f"Training set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}")

# 定义第一阶段的 DNN 模型：估计均值函数 m(x)
def create_first_dnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))  # 输出均值 m(x)
    return model

# 定义第二阶段的 DNN 模型：估计参数 length_scale, sigma_squared, nu
def create_second_dnn_model(input_shape):

    inputs = Input(shape=(input_shape,))

    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    
    params_output = Dense(2, activation='softplus', name='params_output')(x)  # 确保输出为正数
    
    smoothness_output = Dense(2, activation='softmax', name='smoothness_output')(x)
    
    model = Model(inputs=inputs, outputs=[params_output, smoothness_output])
    return model

# 加载和标准化输入数据
scaler_input = StandardScaler()
x_train_normalized = scaler_input.fit_transform(np.hstack((x_train, s_train)))  # 训练集数据
x_test_normalized = scaler_input.transform(np.hstack((x_test, s_test)))  # 测试集数据

# 将 s_train 和 s_test 转换为 TensorFlow 张量
s_train_tensor = tf.constant(s_train, dtype=tf.float32)
s_test_tensor = tf.constant(s_test, dtype=tf.float32)

# 定义全局变量
current_length_scale = tf.Variable(1.0, dtype=tf.float32)
current_sigma_squared = tf.Variable(1.0, dtype=tf.float32)
current_nu = tf.Variable(0.5, dtype=tf.float32)

#自定义 GLS 损失函数
def custom_gls_loss(y_true, y_pred, s_batch):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    s_batch = tf.cast(s_batch, tf.float32)
    batch_size = tf.shape(y_true)[0]
    
    residuals = tf.reshape(y_true - y_pred, [batch_size, -1])  # [batch_size, 1]
    current_para = [current_length_scale, current_sigma_squared, current_nu]
    
    # 协方差矩阵
    cov_matrix_batch = matern_covariance(s_batch, current_para)
    precision_matrix_batch = tf.linalg.pinv(cov_matrix_batch)  # 使用广义逆，得到 [batch_size, batch_size]

    
    # 计算 GLS 损失
    intermediate = tf.matmul(precision_matrix_batch, residuals)  # [batch_size, 1]
    gls_loss_matrix = tf.matmul(tf.transpose(residuals), intermediate)  # [1, 1]
    total_gls_loss = gls_loss_matrix[0, 0] / tf.cast(batch_size, tf.float32)

    return total_gls_loss

# 设置批量大小
batch_size = 64

# 创建训练数据集，确保每次都使用相同的批次
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_normalized, y_train, s_train))
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_iterator = iter(train_dataset)
x_batch_first, y_batch_first, s_batch_first = next(train_iterator)

# 将批次数据转换为 float32 类型
x_batch_first = tf.cast(x_batch_first, tf.float32)
y_batch_first = tf.cast(y_batch_first, tf.float32)
s_batch_first = tf.cast(s_batch_first, tf.float32)

# 定义参数范围
length_scale_range = np.linspace(0.1, 100, 20)
sigma_squared_range = np.linspace(0.1, 100, 20)
smoothness_range = [0.5]

# 创建参数网格
parameter_grid = list(product(length_scale_range, sigma_squared_range, smoothness_range))

# 设置训练轮数
epochs = 20
epochs_second = 80

# 初始化最优结果
best_loss = float('inf')
best_params = None
best_y_pred_test = None
best_first_model = None  # 新增：用于保存最优的第一阶段模型

# 定义 GLS 损失函数用于评估
def gls_loss(y_true, y_pred, precision_matrix):
    residuals = tf.reshape(y_true - y_pred, [-1, 1])  # 使用 TensorFlow 的 reshape
    residuals = tf.cast(residuals, tf.float32)
    precision_matrix = tf.cast(precision_matrix, tf.float32)  # 将 precision_matrix 转换为 float32
    gls_loss_value = tf.matmul(tf.transpose(residuals), tf.matmul(precision_matrix, residuals))  # 使用 tf.matmul 进行矩阵乘法
    return gls_loss_value[0, 0] / tf.cast(tf.shape(y_true)[0], tf.float32)  # 使用 TensorFlow 的 cast 将样本数转换为 float32

# 测试集数据
x_test_batch = tf.constant(x_test_normalized, dtype=tf.float32)
s_test_batch = tf.constant(s_test, dtype=tf.float32)

# 训练集数据（用于后续在最优参数下计算训练集损失）
x_train_batch = tf.constant(x_train_normalized, dtype=tf.float32)
s_train_batch_full = tf.constant(s_train, dtype=tf.float32)
y_train_full = tf.constant(y_train.flatten(), dtype=tf.float32)

# 创建日志文件
log_file = open('training_log.txt', 'w')

# 开始参数网格搜索
for length_scale, sigma_squared, smoothness in parameter_grid:
    log_file.write(f"\nEvaluating parameters: length_scale={length_scale}, sigma_squared={sigma_squared}, smoothness={smoothness}\n")
    print(f"\nEvaluating parameters: length_scale={length_scale}, sigma_squared={sigma_squared}, smoothness={smoothness}")
    
    # 更新全局变量
    current_length_scale.assign(length_scale)
    current_sigma_squared.assign(sigma_squared)
    current_nu.assign(smoothness)
    
    # 创建并训练第一阶段模型
    first_model = create_first_dnn_model(x_train_normalized.shape[1])
    optimizer_first = tf.keras.optimizers.Adam()
    
    # 记录第一阶段的损失
    first_stage_losses = []
    
    for epoch in range(epochs):
       #epoch=1
        with tf.GradientTape() as tape:
            y_pred = first_model(x_batch_first, training=True)#64*1
            #y_batch_first 64*1  
            #s_batch_first 64*1
            loss = custom_gls_loss(y_batch_first, y_pred, s_batch_first)
        gradients = tape.gradient(loss, first_model.trainable_variables)
        optimizer_first.apply_gradients(zip(gradients, first_model.trainable_variables))
        first_stage_losses.append(loss.numpy())
        
        # 打印或记录训练损失
        print(f"First Stage Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}")
        log_file.write(f"First Stage Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}\n")
    
    # 第一阶段预测残差
    y_pred_train = first_model(x_batch_first, training=False).numpy().flatten()
    residuals_train = y_batch_first.numpy().flatten() - y_pred_train
    
    # 第二阶段：使用残差和 x(s) 输入，估计 length_scale, sigma_squared, smoothness
    training_fields = np.hstack((residuals_train.reshape(-1, 1), x_batch_first.numpy()))
    s_train_batch = s_batch_first.numpy()
    training_fields_with_s = np.hstack((training_fields, s_train_batch))
    
    # 标准化第二阶段输入数据
    scaler_second_input = StandardScaler()
    training_fields_normalized = scaler_second_input.fit_transform(training_fields_with_s)
    
    # 标签为当前的 length_scale, sigma_squared, smoothness
    labels = np.array([[length_scale, sigma_squared, smoothness]] * len(training_fields_normalized))
    
    # 标准化标签
    scaler_labels = StandardScaler()
    labels_normalized = scaler_labels.fit_transform(labels)
    
    # 对于 smoothness，提取原始值用于分类标签
    smoothness_labels = np.where(labels[:, 2] == 0.5, 0, 1)
    
    # 将训练数据转换为 TensorFlow 张量
    training_fields_normalized_tensor = tf.constant(training_fields_normalized, dtype=tf.float32)
    labels_params_tensor = tf.constant(labels_normalized[:, :2], dtype=tf.float32)
    smoothness_labels_tensor = tf.constant(smoothness_labels, dtype=tf.int32)
    
    # 创建并训练第二阶段模型
    second_model = create_second_dnn_model(training_fields_normalized.shape[1])
    second_model.compile(optimizer=Adam(learning_rate=0.001),
                         loss={'params_output': 'mse',
                               'smoothness_output': 'sparse_categorical_crossentropy'})
    
    # 记录第二阶段的损失
    second_stage_losses = []
    
    # 自定义回调函数记录每个 epoch 的损失
    class LossHistory(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            loss = logs.get('loss')
            second_stage_losses.append(loss)
            # 打印或记录损失
            print(f"Second Stage Epoch {epoch+1}/{epochs_second}, Loss: {loss}")
            log_file.write(f"Second Stage Epoch {epoch+1}/{epochs_second}, Loss: {loss}\n")
    
    loss_history = LossHistory()
    
    second_model.fit(training_fields_normalized_tensor,
                     {'params_output': labels_params_tensor, 'smoothness_output': smoothness_labels_tensor},
                     batch_size=batch_size,
                     epochs=epochs_second,
                     verbose=0,
                     callbacks=[loss_history])
    
    # 在测试集上评估
    y_pred_test = first_model(x_test_batch, training=False).numpy().flatten()
    residuals_test = y_test.flatten() - y_pred_test
  
    
    test_fields = np.hstack((residuals_test.reshape(-1, 1), x_test_batch.numpy()))
    test_fields_with_s = np.hstack((test_fields, s_test_batch.numpy()))
    test_fields_normalized = scaler_second_input.transform(test_fields_with_s)
    
    test_fields_normalized_tensor = tf.constant(test_fields_normalized, dtype=tf.float32)
    
    # 预测参数
    estimated_params_normalized, smoothness_predictions = second_model.predict(test_fields_normalized_tensor)
    estimated_params = scaler_labels.inverse_transform(np.hstack((estimated_params_normalized, np.zeros((estimated_params_normalized.shape[0], 1)))))
    
    # 分别获取 length_scale 和 sigma_squared，并确保为正数
    estimated_length_scale = np.maximum(estimated_params[:, 0], 1e-5)
    estimated_sigma_squared = np.maximum(estimated_params[:, 1], 1e-5)
    
    # 处理 smoothness 输出
    predicted_classes = np.argmax(smoothness_predictions, axis=1)
    predicted_smoothness = np.where(predicted_classes == 0, 0.5, 1.5)
    
    # 组合参数的平均值
    cishi_params = (np.mean(estimated_length_scale),
                    np.mean(estimated_sigma_squared),
                    np.mean(predicted_smoothness))
    
    print(f"Estimated parameters: length_scale={cishi_params[0]}, sigma_squared={cishi_params[1]}, smoothness={cishi_params[2]}")
    log_file.write(f"Estimated parameters: length_scale={cishi_params[0]}, sigma_squared={cishi_params[1]}, smoothness={cishi_params[2]}\n")
    
 # 使用 GLS 损失计算测试集损失
    try:
    # 在测试集上计算协方差矩阵和精度矩阵
       cov_matrix_test = matern_covariance(s_test_batch, cishi_params)
       precision_matrix_test = tf.linalg.pinv(cov_matrix_test)  # 使用广义逆
    
    # 计算 GLS 损失
       y_test_tensor = tf.constant(y_test.flatten(), dtype=tf.float32)
       y_pred_test_tensor = tf.constant(y_pred_test, dtype=tf.float32)
       loss = gls_loss(y_test_tensor, y_pred_test_tensor, precision_matrix_test)
       print(f"Current GLS loss on test set: {loss.numpy()}")
       log_file.write(f"Current GLS loss on test set: {loss.numpy()}\n")

        # 更新最优结果
       if loss.numpy() < best_loss:
            best_loss = loss.numpy()
            best_params = (length_scale, sigma_squared, smoothness)
            best_y_pred_test = y_pred_test
            best_first_model = first_model  # 保存当前最优的第一阶段模型
            best_precision_matrix_train = None  # 将在后面计算
            best_cishi_params = cishi_params  # 保存估计的参数
    except tf.errors.InvalidArgumentError as e:
        print("Skipping invalid precision matrix during GLS loss computation")
        log_file.write("Skipping invalid precision matrix during GLS loss computation\n")
        print(f"Error message: {e}")
        log_file.write(f"Error message: {e}\n")
        continue

   
# 关闭日志文件
log_file.close()

if best_params is not None:
    print(f"\nBest parameters found: length_scale={best_params[0]}, sigma_squared={best_params[1]}, smoothness={best_params[2]}")
    print(f"Best GLS loss on test set: {best_loss}")

# 逆归一化预测值和真实值
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()  # 保持为一维
    y_pred_test_final_inv = scaler_y.inverse_transform(best_y_pred_test.reshape(-1, 1)).flatten()

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

    # 在训练集上计算损失，使用部分训练集（500个样本）
    subset_size = 500
    s_train_subset = s_train_batch_full[:subset_size]
    y_train_subset = y_train_full[:subset_size]
    x_train_subset = x_train_batch[:subset_size]

    # 使用最优的第一阶段模型在训练子集上进行预测
    y_pred_train_subset = best_first_model(x_train_subset, training=False).numpy().flatten()
    residuals_train_subset = y_train_subset.numpy() - y_pred_train_subset

    # 计算协方差矩阵和精度矩阵（在训练子集上）
    try:
        cov_matrix_train_subset = matern_covariance(s_train_subset,best_cishi_params)
        
        precision_matrix_train_subset = tf.linalg.pinv(cov_matrix_train_subset)  # 使用广义逆
      
        # 计算训练子集上的 GLS 损失
        y_train_subset_tensor = y_train_subset  # 已经是 tf.float32 类型
        y_pred_train_subset_tensor = tf.constant(y_pred_train_subset, dtype=tf.float32)
        train_loss = gls_loss(y_train_subset_tensor, y_pred_train_subset_tensor, precision_matrix_train_subset)
        print(f"GLS loss on training (500 samples) with best parameters: {train_loss.numpy()}")
    except tf.errors.InvalidArgumentError as e:
        print("Failed to compute training loss with best parameters due to invalid precision matrix.")
        print(f"Error message: {e}")
else:
    print("No valid parameters found during grid search.")

mean_predicted_y_test =np.mean(best_y_pred_test)
print(f"Best predicted mean on test:{mean_predicted_y_test}")

# 对测试集上的预测值进行逆归一化
best_y_pred_test_inverse = scaler_y.inverse_transform(best_y_pred_test.reshape(-1, 1)).flatten()

# 计算逆归一化后的预测值的均值
mean_actual_predicted_y_test = np.mean(best_y_pred_test_inverse)

print(f"Mean predicted y on test(after inverse transform):{mean_actual_predicted_y_test}")


end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time} seconds")
### 0.1 50 0.5