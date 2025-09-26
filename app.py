import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Streamlit app title
st.title("Linear Regression with Outlier Detection")

# Sidebar controls
st.sidebar.header("Data Parameters")
n = st.sidebar.slider("Number of Data Points (n)", min_value=100, max_value=1000, value=300, step=50)
a = st.sidebar.slider("Coefficient a", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
b = st.sidebar.slider("Intercept b", min_value=-50.0, max_value=50.0, value=5.0, step=1.0)
var = st.sidebar.slider("Noise Variance", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)

# Data generation
np.random.seed(42)  # for reproducibility
x = np.random.uniform(-50, 50, n).reshape(-1, 1)
noise = np.random.normal(0, np.sqrt(var), size=n)
y = a * x.flatten() + b + noise

# Fit linear regression
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Calculate residuals (distance from regression line)
residuals = np.abs(y - y_pred)

# Find top 5 outliers
outlier_indices = residuals.argsort()[-5:][::-1]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, alpha=0.6, label="Data Points")
ax.plot(x, y_pred, color="red", linewidth=2, label="Regression Line")

# Highlight outliers
ax.scatter(x[outlier_indices], y[outlier_indices], color="orange", edgecolors="black", s=120, label="Outliers")
for idx in outlier_indices:
    ax.text(x[idx], y[idx], f"({x[idx][0]:.1f}, {y[idx]:.1f})", fontsize=8, color="black")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Linear Regression with Outlier Detection")
ax.legend()

st.pyplot(fig)

# Display regression results
st.subheader("Regression Results")
st.write(f"Fitted Line: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
st.write("Top 5 Outliers (x, y):")
for idx in outlier_indices:
    st.write(f"({x[idx][0]:.3f}, {y[idx]:.3f}) - Residual: {residuals[idx]:.3f}")
