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
import time

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

# 归一化特征和目标变量
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

scaler_s = MinMaxScaler()
s = scaler_s.fit_transform(s)

scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)

# 手动指定测试集的大小
test_size = 640

# 划分训练集和测试集
X_train, X_test, s_train, s_test, y_train, y_test = train_test_split(X, s, y, test_size=test_size, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Matérn 协方差函数，使用 TensorFlow 操作
def matern_covariance(d, length_scale, sigma_squared, nu):
    d = tf.cast(d, tf.float32)
    length_scale = tf.cast(length_scale, tf.float32)
    sigma_squared = tf.cast(sigma_squared, tf.float32)
    nu = tf.cast(nu, tf.float32)
    
    d = tf.clip_by_value(d, 0, 1e5)
    length_scale = tf.clip_by_value(length_scale, 1e-5, 1e5)
    
    # 定义不同的 nu 值对应的计算函数
    def compute_nu_0_5():
        return sigma_squared * tf.exp(-d / length_scale)
    
    def compute_nu_1_5():
        sqrt_3 = tf.sqrt(3.0)
        sqrt_3_d_theta = sqrt_3 * d / length_scale
        return sigma_squared * (1.0 + sqrt_3_d_theta) * tf.exp(-sqrt_3_d_theta)
    
    def invalid_nu():
        # 返回一个与 d 相同形状的 NaN 张量
        return tf.fill(tf.shape(d), tf.constant(float('nan'), dtype=tf.float32))
    
    # 使用 TensorFlow 控制流 tf.case
    cov = tf.case(
        [
            (tf.equal(nu, 0.5), compute_nu_0_5),
            (tf.equal(nu, 1.5), compute_nu_1_5)
        ],
        default=invalid_nu,
        exclusive=True
    )
    
    return cov

# 生成基于经纬度的协方差矩阵，使用 TensorFlow 操作
def covariance_matrix(length_scale, sigma_squared, smoothness, longitudes, latitudes):
    # 确保所有输入都是 tf.float32 类型的张量
    length_scale = tf.cast(length_scale, tf.float32)
    sigma_squared = tf.cast(sigma_squared, tf.float32)
    smoothness = tf.cast(smoothness, tf.float32)
    longitudes = tf.cast(longitudes, tf.float32)
    latitudes = tf.cast(latitudes, tf.float32)
    
    # 定义有效的 smoothness 处理函数
    def valid_smoothness():
        # 计算位置矩阵
        positions = tf.stack([longitudes, latitudes], axis=1)
        # 计算距离矩阵
        diff = tf.expand_dims(positions, axis=1) - tf.expand_dims(positions, axis=0)
        dist_matrix = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))
        # 计算协方差矩阵
        cov_matrix = matern_covariance(dist_matrix, length_scale, sigma_squared, smoothness)
        # 添加微小值以提高数值稳定性
        jitter = 1e-2  # 增大 jitter 值以提高数值稳定性
        cov_matrix += jitter * tf.eye(tf.shape(cov_matrix)[0])
        return cov_matrix

    # 定义无效的 smoothness 处理函数
    def invalid_smoothness():
        # 返回一个填充 NaN 的矩阵
        nan_matrix = tf.fill([tf.shape(longitudes)[0], tf.shape(longitudes)[0]], tf.constant(float('nan'), dtype=tf.float32))
        return nan_matrix

    # 使用 TensorFlow 控制流 tf.cond
    is_valid_smoothness = tf.reduce_any(tf.equal(smoothness, [0.5, 1.5]))
    cov_matrix = tf.cond(
        is_valid_smoothness,
        true_fn=valid_smoothness,
        false_fn=invalid_smoothness
    )

    return cov_matrix

# 定义第一阶段的 DNN 模型：估计均值函数 m(x)
def create_first_dnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))  # 输出均值 m(x)
    return model

# 定义第二阶段的 DNN 模型：估计参数 length_scale, sigma_squared, smoothness
def create_second_dnn_model(input_shape):
    # 定义输入
    inputs = Input(shape=(input_shape,))
    
    # 公共层
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    
    # length_scale 和 sigma_squared 分支
    params_output = Dense(2, activation='softplus', name='params_output')(x)  # 确保输出为正数
    
    # smoothness 分支
    smoothness_output = Dense(2, activation='softmax', name='smoothness_output')(x)
    
    # 定义模型
    model = Model(inputs=inputs, outputs=[params_output, smoothness_output])
    return model

# 加载和标准化输入数据
scaler_input = StandardScaler()
X_train_normalized = scaler_input.fit_transform(np.hstack((X_train, s_train)))  # 训练集数据
X_test_normalized = scaler_input.transform(np.hstack((X_test, s_test)))  # 测试集数据

# 将 s_train 和 s_test 转换为 TensorFlow 张量
s_train_tensor = tf.constant(s_train, dtype=tf.float32)
s_test_tensor = tf.constant(s_test, dtype=tf.float32)

# 定义全局变量
current_length_scale = tf.Variable(1.0, dtype=tf.float32)
current_sigma_squared = tf.Variable(1.0, dtype=tf.float32)
current_smoothness = tf.Variable(0.5, dtype=tf.float32)

# 自定义 GLS 损失函数，接收 s_batch 作为参数
def custom_gls_loss(y_true, y_pred, s_batch):
    # 将张量转换为 float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    s_batch = tf.cast(s_batch, tf.float32)

    batch_size = tf.shape(y_true)[0]
    output_dim = tf.shape(y_true)[1]

    # 计算残差
    residuals = y_true - y_pred  # 形状：[batch_size, output_dim]

    # 计算协方差矩阵
    cov_matrix_batch = covariance_matrix(
        current_length_scale,
        current_sigma_squared,
        current_smoothness,
        s_batch[:, 0],
        s_batch[:, 1]
    )

    # 计算精度矩阵
    L = tf.linalg.cholesky(cov_matrix_batch)
    precision_matrix_batch = tf.linalg.cholesky_solve(L, tf.eye(batch_size, dtype=tf.float32))

    # 矢量化计算 GLS 损失
    intermediate = tf.matmul(precision_matrix_batch, residuals)
    gls_loss_matrix = tf.matmul(tf.transpose(residuals), intermediate)
    gls_loss_values = tf.linalg.diag_part(gls_loss_matrix)
    total_gls_loss = tf.reduce_mean(gls_loss_values) / tf.cast(batch_size, tf.float32)

    return total_gls_loss

# 设置批量大小
batch_size = 64

# 创建训练数据集，确保每次都使用相同的批次
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_normalized, y_train, s_train))
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_iterator = iter(train_dataset)
x_batch_first, y_batch_first, s_batch_first = next(train_iterator)

# 将批次数据转换为 float32 类型
x_batch_first = tf.cast(x_batch_first, tf.float32)
y_batch_first = tf.cast(y_batch_first, tf.float32)
s_batch_first = tf.cast(s_batch_first, tf.float32)

# 定义参数范围
length_scale_range = [0.1]#np.linspace(0.1, 100, 100)
sigma_squared_range = np.linspace(0.1, 100, 100)
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
x_test_batch = tf.constant(X_test_normalized, dtype=tf.float32)
s_test_batch = tf.constant(s_test, dtype=tf.float32)

# 训练集数据（用于后续在最优参数下计算训练集损失）
x_train_batch = tf.constant(X_train_normalized, dtype=tf.float32)
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
    current_smoothness.assign(smoothness)
    
    # 创建并训练第一阶段模型
    first_model = create_first_dnn_model(X_train_normalized.shape[1])
    optimizer_first = tf.keras.optimizers.Adam()
    
    # 记录第一阶段的损失
    first_stage_losses = []
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = first_model(x_batch_first, training=True)
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
        cov_matrix_test = covariance_matrix(cishi_params[0], cishi_params[1], cishi_params[2],
                                            s_test_batch[:, 0], s_test_batch[:, 1])
        L_test = tf.linalg.cholesky(cov_matrix_test)
        precision_matrix_test = tf.linalg.cholesky_solve(L_test, tf.eye(len(y_test), dtype=tf.float32))
    
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
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_test_final_inv = scaler_y.inverse_transform(best_y_pred_test )

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
        cov_matrix_train_subset = covariance_matrix(best_cishi_params[0], best_cishi_params[1], best_cishi_params[2],
                                                    s_train_subset[:, 0], s_train_subset[:, 1])
        L_train_subset = tf.linalg.cholesky(cov_matrix_train_subset)
        precision_matrix_train_subset = tf.linalg.cholesky_solve(L_train_subset, tf.eye(subset_size, dtype=tf.float32))
        
        # 计算训练子集上的 GLS 损失
        y_train_subset_tensor = y_train_subset  # 已经是 tf.float32 类型
        y_pred_train_subset_tensor = tf.constant(y_pred_train_subset, dtype=tf.float32)
        train_loss = gls_loss(y_train_subset_tensor, y_pred_train_subset_tensor, precision_matrix_train_subset)
        print(f"GLS loss on training subset (500 samples) with best parameters: {train_loss.numpy()}")
    except tf.errors.InvalidArgumentError as e:
        print("Failed to compute training loss with best parameters due to invalid precision matrix.")
        print(f"Error message: {e}")
else:
    print("No valid parameters found during grid search.")

mean_predicted_y_test =np.mean(best_y_pred_test)
print(f"Best predicted mean function on test set:{mean_predicted_y_test}")

# 对测试集上的预测值进行逆归一化
best_y_pred_test_inverse = scaler_y.inverse_transform(best_y_pred_test.reshape(-1, 1)).flatten()

# 计算逆归一化后的预测值的均值
mean_actual_predicted_y_test = np.mean(best_y_pred_test_inverse)

print(f"Mean of predicted y on test set (after inverse transform):{mean_actual_predicted_y_test}")


end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time} seconds")

#Best parameters found: length_scale=0.1, sigma_squared=8.9, smoothness=0.5
#Best GLS loss on test set: 0.09269507229328156
#Best predicted mean function on test set:0.14408168
#Total runtime: 2857.3849103450775 seconds

#(0.1,100,100) (0.1,100,100),0.5
#Best parameters: 
#length_scale=0.1, sigma_squared=71.74545454545454, smoothness=0.5
#Best GLS loss on test set: 0.025596076622605324
#GLS loss on training subset (500 samples): 0.02037927508354187
#Best predicted mean function on test set:0.1352233588695526
#Mean of predicted y on test set (after inverse transform):80582.6015625
#Total runtime: 15176.581571817398 seconds