# ============================================
# SALES FORECAST MODEL - Complete Project
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: SAMPLE DATA CREATE
# ============================================

np.random.seed(42)

# 3 years monthly sales data
dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='MS')
months = len(dates)

# Realistic sales pattern: trend + seasonality + noise
trend = np.linspace(50000, 95000, months)
seasonality = 8000 * np.sin(2 * np.pi * np.arange(months) / 12)
noise = np.random.normal(0, 3000, months)
sales = trend + seasonality + noise

df = pd.DataFrame({
    'Date': dates,
    'Sales': sales.astype(int)
})

print("=" * 50)
print("SALES DATA - First 12 Months:")
print("=" * 50)
print(df.head(12).to_string(index=False))
print(f"\nTotal Records: {len(df)}")
print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Average Monthly Sales: Rs.{df['Sales'].mean():,.0f}")
print(f"Max Sales: Rs.{df['Sales'].max():,.0f}")
print(f"Min Sales: Rs.{df['Sales'].min():,.0f}")

# ============================================
# STEP 2: FEATURE ENGINEERING
# ============================================

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Month_Num'] = range(1, len(df) + 1)
df['Quarter'] = df['Date'].dt.quarter
df['Sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12)

print("\n" + "=" * 50)
print("FEATURE ENGINEERING COMPLETE:")
print("=" * 50)
print("Features added: Month, Year, Month_Num, Quarter,")
print("                Sin_Month, Cos_Month")

# ============================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================

# Last 6 months used for testing
train_size = len(df) - 6
train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = ['Month_Num', 'Sin_Month', 'Cos_Month', 'Quarter']
X_train = train[features]
y_train = train['Sales']
X_test = test[features]
y_test = test['Sales']

print("\n" + "=" * 50)
print("TRAIN/TEST SPLIT:")
print("=" * 50)
print(f"Training Data: {len(train)} months")
print(f"Testing Data:  {len(test)} months")

# ============================================
# STEP 4: MODEL TRAINING
# ============================================

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

train_pred = model.predict(X_train_poly)
test_pred = model.predict(X_test_poly)

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETE!")
print("=" * 50)
print("Model: Polynomial Regression (Degree 2)")

# ============================================
# STEP 5: MODEL EVALUATION
# ============================================

mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100

print("\n" + "=" * 50)
print("MODEL PERFORMANCE:")
print("=" * 50)
print(f"MAE  (Mean Absolute Error):  Rs.{mae:,.0f}")
print(f"RMSE (Root Mean Sq Error):   Rs.{rmse:,.0f}")
print(f"MAPE (Mean Abs % Error):     {mape:.2f}%")
print(f"Accuracy:                    {100-mape:.2f}%")

# ============================================
# STEP 6: FUTURE FORECAST (Next 6 months)
# ============================================

future_dates = pd.date_range(start='2025-01-01', periods=6, freq='MS')
future_month_num = range(len(df) + 1, len(df) + 7)
future_months = future_dates.month

future_df = pd.DataFrame({
    'Month_Num': future_month_num,
    'Sin_Month': np.sin(2 * np.pi * future_months / 12),
    'Cos_Month': np.cos(2 * np.pi * future_months / 12),
    'Quarter': (future_months - 1) // 3 + 1
})

future_poly = poly.transform(future_df[features])
future_pred = model.predict(future_poly)

print("\n" + "=" * 50)
print("FUTURE SALES FORECAST (Jan-Jun 2025):")
print("=" * 50)
for date, pred in zip(future_dates, future_pred):
    print(f"  {date.strftime('%B %Y')}: Rs.{pred:>10,.0f}")

# ============================================
# STEP 7: VISUALIZATION
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Sales Forecast Dashboard', fontsize=18, fontweight='bold', y=0.98)
fig.patch.set_facecolor('#F8F9FA')

ax1 = axes[0, 0]
ax1.set_facecolor('#FFFFFF')
ax1.plot(train['Date'], train['Sales'], color='#2196F3', linewidth=2, label='Training Data')
ax1.plot(test['Date'], test['Sales'], color='#4CAF50', linewidth=2, label='Actual (Test)')
ax1.plot(test['Date'], test_pred, color='#FF9800', linewidth=2, linestyle='--', label='Predicted')
ax1.plot(future_dates, future_pred, color='#E91E63', linewidth=2.5,
         linestyle='--', marker='o', markersize=6, label='Future Forecast')
ax1.axvline(x=test['Date'].iloc[0], color='gray', linestyle=':', alpha=0.7, label='Train/Test Split')
ax1.set_title('Sales Forecast Overview', fontweight='bold', fontsize=12)
ax1.set_xlabel('Date')
ax1.set_ylabel('Sales (Rs.)')
ax1.legend(fontsize=8)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.set_facecolor('#FFFFFF')
ax2.scatter(y_test, test_pred, color='#673AB7', alpha=0.8, s=80, edgecolors='white', linewidth=1.5)
min_val = min(y_test.min(), test_pred.min())
max_val = max(y_test.max(), test_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_title('Actual vs Predicted Sales', fontweight='bold', fontsize=12)
ax2.set_xlabel('Actual Sales (Rs.)')
ax2.set_ylabel('Predicted Sales (Rs.)')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.set_facecolor('#FFFFFF')
monthly_avg = df.groupby('Month')['Sales'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
colors = ['#FF6B6B' if v == monthly_avg.max() else '#4ECDC4' for v in monthly_avg.values]
bars = ax3.bar(month_names, monthly_avg.values, color=colors, edgecolor='white', linewidth=1.5)
ax3.set_title('Average Sales by Month (Seasonality)', fontweight='bold', fontsize=12)
ax3.set_xlabel('Month')
ax3.set_ylabel('Average Sales (Rs.)')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
ax3.grid(True, alpha=0.3, axis='y')
peak_month = monthly_avg.idxmax()
ax3.annotate(f'Peak: {month_names[peak_month-1]}',
             xy=(peak_month-1, monthly_avg.max()),
             xytext=(peak_month+1, monthly_avg.max()),
             fontsize=9, color='#FF6B6B', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#FF6B6B'))

ax4 = axes[1, 1]
ax4.set_facecolor('#FFFFFF')
future_month_names = [d.strftime('%b %Y') for d in future_dates]
bar_colors = ['#3F51B5', '#2196F3', '#03A9F4', '#00BCD4', '#009688', '#4CAF50']
bars2 = ax4.bar(future_month_names, future_pred, color=bar_colors,
                edgecolor='white', linewidth=1.5)
ax4.set_title('Future Sales Forecast (Jan-Jun 2025)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Month')
ax4.set_ylabel('Forecasted Sales (Rs.)')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
ax4.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=30)
for bar, val in zip(bars2, future_pred):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
             f'Rs.{val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('sales_forecast_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
plt.show()

print("\n" + "=" * 50)
print("Dashboard saved: sales_forecast_dashboard.png")
print("Project Complete!")
print("=" * 50)
print("\nBUSINESS INSIGHTS:")
print(f"  Average Monthly Growth: Rs.{(df['Sales'].iloc[-1] - df['Sales'].iloc[0]) / len(df):,.0f}")
print(f"  Best Month Pattern: {month_names[monthly_avg.idxmax()-1]} (Peak Season)")
print(f"  6-Month Forecast Total: Rs.{sum(future_pred):,.0f}")
print(f"  Model Accuracy: {100-mape:.1f}%")
