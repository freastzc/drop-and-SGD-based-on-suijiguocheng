import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Flatten
import yfinance as yf

# Step 1: 获取和预处理数据
ticker = "AAPL"  # 苹果公司股票
data = pd.read_csv("financial_data.csv")
prices = data['收盘'].values  # 使用收盘价

# 数据归一化
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# 生成时间序列数据
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(prices_scaled, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据可视化
plt.figure(figsize=(10, 6))
plt.plot(prices, label='Original Prices')
plt.title(f'{ticker} Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Step 2: 布朗运动层定义
class BrownianMotionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BrownianMotionLayer, self).__init__()
        self.units = units

    def call(self, inputs):
        shape = tf.shape(inputs)
        delta_t = 1 / tf.cast(shape[1], tf.float32)
        # 添加更强的波动：模拟更剧烈的市场波动
        noise = tf.random.normal(shape) * 0.05  # 增大噪声的尺度
        bm = tf.cumsum(noise * tf.sqrt(delta_t), axis=1)
        return inputs + bm

# Step 3: 在 CNN 模型中加入布朗运动层
def create_cnn_with_brownian_motion(input_shape):
    input_layer = Input(shape=input_shape)
    bm_layer = BrownianMotionLayer(units=input_shape[0])(input_layer)  # 在卷积前加入布朗运动
    conv_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(bm_layer)
    flatten_layer = Flatten()(conv_layer)
    dense_layer = Dense(50, activation='relu')(flatten_layer)
    output_layer = Dense(1)(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Step 4: 传统 CNN 模型
def create_simple_cnn(input_shape):
    input_layer = Input(shape=input_shape)
    conv_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
    flatten_layer = Flatten()(conv_layer)
    dense_layer = Dense(50, activation='relu')(flatten_layer)
    output_layer = Dense(1)(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Step 5: 训练模型
cnn_model = create_simple_cnn((seq_length, 1))
cnn_with_bm_model = create_cnn_with_brownian_motion((seq_length, 1))

# 训练 CNN 模型
cnn_history = cnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
# 训练 带有布朗运动层的 CNN 模型
cnn_with_bm_history = cnn_with_bm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Step 6: 模型评估与预测
# 评估模型
cnn_loss, cnn_mae = cnn_model.evaluate(X_test, y_test, verbose=0)
cnn_with_bm_loss, cnn_with_bm_mae = cnn_with_bm_model.evaluate(X_test, y_test, verbose=0)

print(f"传统 CNN 模型 - 测试集 MSE: {cnn_loss:.4f}, MAE: {cnn_mae:.4f}")
print(f"带有布朗运动的 CNN 模型 - 测试集 MSE: {cnn_with_bm_loss:.4f}, MAE: {cnn_with_bm_mae:.4f}")

# Step 7: 预测结果与可视化
cnn_predictions = cnn_model.predict(X_test)
cnn_with_bm_predictions = cnn_with_bm_model.predict(X_test)

# 实际值与预测值可视化
plt.figure(figsize=(14, 7))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='True Prices', color='blue')
plt.plot(scaler.inverse_transform(cnn_predictions), label='Traditional CNN Predictions', color='green')
plt.plot(scaler.inverse_transform(cnn_with_bm_predictions), label='CNN with Brownian Motion Predictions', color='orange')
plt.title('Model Predictions vs True Prices', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 预测误差分布
cnn_errors = scaler.inverse_transform(y_test.reshape(-1, 1)) - scaler.inverse_transform(cnn_predictions)
cnn_with_bm_errors = scaler.inverse_transform(y_test.reshape(-1, 1)) - scaler.inverse_transform(cnn_with_bm_predictions)

plt.figure(figsize=(14, 7))
plt.hist(cnn_errors, bins=50, alpha=0.7, label='CNN Errors', color='green')
plt.hist(cnn_with_bm_errors, bins=50, alpha=0.7, label='CNN with BM Errors', color='orange')
plt.title('Prediction Error Distribution', fontsize=16)
plt.xlabel('Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Step 8: 可视化训练过程损失曲线
plt.figure(figsize=(12, 6))
plt.plot(cnn_history.history['loss'], label='CNN Model Train Loss', color='green')
plt.plot(cnn_history.history['val_loss'], label='CNN Model Validation Loss', color='blue')
plt.plot(cnn_with_bm_history.history['loss'], label='CNN with BM Model Train Loss', color='orange')
plt.plot(cnn_with_bm_history.history['val_loss'], label='CNN with BM Model Validation Loss', color='red')
plt.title('Training and Validation Loss Comparison', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
