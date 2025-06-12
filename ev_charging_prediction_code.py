
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv('ev_charging_demand.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Feature selection and normalization
features = ['charging_sessions', 'temperature', 'day_of_week', 'is_holiday']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Create sequences for LSTM
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, 1:])  # use features except target
        y.append(data[i+seq_length, 0])     # target is 'charging_sessions'
    return np.array(X), np.array(y)

seq_len = 24
X, y = create_sequences(scaled_data, seq_length=seq_len)

# Split data
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_len, X.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(
    np.hstack((y_pred.reshape(-1, 1), X_test[:, -1, :]))
)[:, 0]
y_test_rescaled = scaler.inverse_transform(
    np.hstack((y_test.reshape(-1, 1), X_test[:, -1, :]))
)[:, 0]

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.title('EV Charging Demand Prediction')
plt.xlabel('Time (hours)')
plt.ylabel('Charging Sessions')
plt.legend()
plt.tight_layout()
plt.savefig('ev_demand_prediction.png')
plt.show()
