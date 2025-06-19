# Edunet_InternshipProject

# EV Charging Demand Prediction using LSTM

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network to predict Electric Vehicle (EV) charging demand based on historical data and environmental factors. The model takes into account features like temperature, day of the week, and holiday status to forecast the number of charging sessions.

## Features

- Time-series forecasting using LSTM neural networks
- Multi-feature input (charging sessions, temperature, day of week, holiday status)
- Data normalization using MinMaxScaler
- Sequence creation for time-series data
- Model evaluation and visualization

## Requirements

- Python 3.7+
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - tensorflow

Install requirements with:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## Dataset

The model expects a CSV file named `ev_charging_demand.csv` with the following columns:
- `timestamp`: DateTime index of the observations
- `charging_sessions`: Number of charging sessions (target variable)
- `temperature`: Temperature in degrees Celsius
- `day_of_week`: Day of week (0-6 where 0 is Monday)
- `is_holiday`: Binary indicator for holidays (1 if holiday, 0 otherwise)

## Usage

1. Place your `ev_charging_demand.csv` file in the same directory as the script
2. Run the script:
```bash
python ev_charging_prediction_code.py
```

## Output

The script will:
1. Train an LSTM model on 80% of the data
2. Test the model on the remaining 20%
3. Generate a plot comparing actual vs predicted values
4. Save the plot as `ev_demand_prediction.png`
