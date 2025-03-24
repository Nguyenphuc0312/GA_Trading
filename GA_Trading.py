import yfinance as yf
import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

#Tải dữ liệu chứng khoán
def fetch_data(ticker='AAPL', period='1y', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    data.dropna(inplace=True)
    return data

#Tính RSI (đo động lượng giá, cho biết khi nào bị mua/bán quá mức)
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

#Tính MACD và đường tín hiệu 
def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Tính Bollinger Bands
def compute_bollinger_bands(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + (std * 2), sma - (std * 2)

# Tính toán tất cả chỉ báo và đưa vào tập dữ liệu
def calculate_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['MACD'], data['Signal'] = compute_macd(data['Close'])
    data['Upper_BB'], data['Lower_BB'] = compute_bollinger_bands(data['Close'])
    data.dropna(inplace=True)
    return data

#====================Xây dựng thuậth toán di truyền==========================
# Lấy dữ liệu chứng khoán
data = fetch_data('AAPL', '1y', '1d')
data = calculate_indicators(data)

# Khởi tạo thuật toán di truyền
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Sinh cá thể
def create_individual():
    return [
        random.uniform(30, 70),   # RSI Threshold
        random.uniform(-2, 2),    # MACD Threshold
        random.uniform(0.8, 1.2), # SMA20/SMA50 Ratio
        random.uniform(0.8, 1.2)  # Bollinger Bands Threshold (Cố định khoảng hợp lý)
    ]
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Hàm đánh giá chiến lược giao dịch
def evaluate(individual):
    capital = 1000
    position = 0  # 0: Không nắm giữ, 1: Đang nắm giữ cổ phiếu
    trades = []
    
    for date, row in data.iterrows():
        rsi, macd, signal, sma20, sma50, upper_bb, lower_bb, close = row[['RSI', 'MACD', 'Signal', 'SMA_20', 'SMA_50', 'Upper_BB', 'Lower_BB', 'Close']]
        rsi_thresh, macd_thresh, sma_thresh, bb_thresh = individual
        
        buy_condition = (rsi < rsi_thresh) and (macd > macd_thresh) and (sma20 > sma50 * sma_thresh)
        sell_condition = (close >= upper_bb * bb_thresh) or (rsi > 100 - rsi_thresh) or (macd < signal)

        if position == 0 and buy_condition:
            position = capital / close
            capital = 0
            trades.append(f"{date.strftime('%Y-%m-%d')} - MUA ở giá {close:.2f}")
        elif position > 0 and sell_condition:
            capital = position * close
            position = 0
            trades.append(f"{date.strftime('%Y-%m-%d')} - BÁN ở giá {close:.2f}")
    
    return (capital if position == 0 else position * data.iloc[-1]['Close'], trades)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Chạy thuật toán di truyền với thông số tối ưu
population = toolbox.population(n=30)
NGEN, CXPB, MUTPB = 20, 0.6, 0.3

best_fitness_values = []
avg_fitness_values = []

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = (fit[0],)
    
    population = toolbox.select(offspring, k=len(population))
    best = max(fits, key=lambda x: x[0])
    avg = sum(f[0] for f in fits) / len(fits)
    best_fitness_values.append(best[0])
    avg_fitness_values.append(avg)
    
    print(f"Generation {gen+1}: Best = {best[0]:.2f}, Avg = {avg:.2f}")

best_ind = tools.selBest(population, k=1)[0]
best_profit, best_trades = evaluate(best_ind)

print("\n==== CHIẾN LƯỢC GIAO DỊCH TỐT NHẤT ====")
print(f"RSI Threshold: {best_ind[0]:.2f} (Mua nếu RSI < {best_ind[0]:.2f}, Bán nếu RSI > {100 - best_ind[0]:.2f})")
print(f"MACD Threshold: {best_ind[1]:.2f} (Mua nếu MACD > {best_ind[1]:.2f}, Bán nếu MACD < Đường tín hiệu)")
print(f"SMA20/SMA50 Ratio: {best_ind[2]:.2f} (Mua nếu SMA20 > SMA50 x {best_ind[2]:.2f})")
print(f"Bollinger Bands Threshold: {best_ind[3]:.2f} (Bán nếu Close >= Upper BB x {best_ind[3]:.2f})")
print(f"Expected Profit: {evaluate(best_ind)[0]:.2f}")

print("\n==== DANH SÁCH GIAO DỊCH ====")
for trade in best_trades:
    print(trade)

plt.figure(figsize=(10, 5))
plt.plot(range(1, NGEN + 1), best_fitness_values, label="Best Fitness", marker='o', linestyle='-')
plt.plot(range(1, NGEN + 1), avg_fitness_values, label="Average Fitness", marker='s', linestyle='--')
plt.xlabel("Generation")
plt.ylabel("Fitness (Profit)")
plt.title("Evolution of Trading Strategy Performance")
plt.legend()
plt.grid(True)
plt.show()
