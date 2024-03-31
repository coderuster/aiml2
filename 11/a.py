import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# i) Create a 2-D array with 3 rows and 4 columns using numpy
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Display shape, itemsize, and datatype of the array
print("Shape of the array:", arr_2d.shape)
print("Size of each element in bytes:", arr_2d.itemsize)
print("Datatype of the array:", arr_2d.dtype)

# Reshape the array as 4 rows and 3 columns
arr_reshaped = arr_2d.reshape(4, 3)
print("Reshaped array:")
print(arr_reshaped)
print()

# ii) Create a 1-D array named profit and sales, then calculate Profit Margin Ratio
profit = np.array([100, 200, 300, 400, 500])
sales = np.array([1000, 1500, 2000, 2500, 3000])
profit_margin_ratio = profit / sales
print("Profit Margin Ratio:")
print(profit_margin_ratio)
print()

# iii) Use matplotlib library to plot a graph by taking any random set of values for x & y
x = np.linspace(0, 10, 50)  # 50 evenly spaced values from 0 to 10
y = np.sin(x)  # Calculate sine of x values
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sine wave')
plt.show()

# iv) Reading any CSV file and storing in a dataframe
df = pd.read_csv('example.csv')  # Replace 'example.csv' with your CSV file
print("\nDataframe from CSV file:")
print(df)
print()

# v) Use matplotlib library to plot a scatter plot with two different classes specifying different color for classes
# Generating random data for demonstration
np.random.seed(0)
x1 = np.random.randn(50)
y1 = np.random.randn(50)
x2 = np.random.randn(50) + 3  # Add offset to create two different classes
y2 = np.random.randn(50) + 3

plt.scatter(x1, y1, color='red', label='Class 1')
plt.scatter(x2, y2, color='blue', label='Class 2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter plot with two classes')
plt.legend()
plt.show()
