import matplotlib.pyplot as plt
#
# # Example data: Replace these lists with your actual data points
x_values = list(range(10, 101, 10))  # This represents the x-axis values

#icarl_dm = [84.8, 58.85, 44.33, 37.55, 32.8, 31.63, 27.87, 26.15, 24.11 ,22.56]
icarl_dsa = [86.6,56.22,43.4, 36.5, 31.15, 29, 28.4, 26.23, 24.33,22.81]
#icarl_dsa_resnet18 = [86.4, 55.5, 47.5 ,38.5, 35, 29, 27.6, 25.1, 23.7, 21.5]  # Replace with your 'dm_convnet' data
icarl = [85.1, 55.55, 45.3, 38.38, 32.14, 29.67, 29.26, 26.33, 24.09, 23.0]  # Replace with your 'icarl' data

# Create the plot
plt.figure(figsize=(10, 3))

# Plot the lines
plt.plot(x_values, icarl_dsa, label='icarl_dsa', color='green')
#plt.plot(x_values, icarl_dsa_resnet18, label='icarl_dsa_resnet18', color='purple')
plt.plot(x_values, icarl, label='icarl', color='red')

# Add a title and labels
plt.title('Comparison of Continual Learning Methods')
plt.xlabel('Tasks')
plt.ylabel('Accuracy')

# Show the legend
plt.legend()
plt.grid()

# Show the plot
plt.show()


# matrix = [
#     [85, 0, 0, 0, 0, 0 , 0 , 0, 0 , 0],
#     [20.5, 85, 0, 0, 0, 0 , 0 , 0, 0 , 0],
#     [0, 32.1, 85, 0, 0, 0 , 0 , 0, 0 , 0],
#     [0, 1.2, 39.5, 85, 0, 0 , 0 , 0, 0 , 0],
#     [0, 0, 0, 32.4, 85, 0 , 0 , 0, 0 , 0],
#     [0, 0, 0, 0, 38.8, 85, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 46.1, 85, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 29.1, 85, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 29.2, 85, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 47, 85],
# ]
#
# for row in matrix:
#     print(' '.join(map(str, row)))

# import numpy as np

# matrix = np.array([
#     [85, 0, 0, 0, 0, 0 , 0 , 0, 0 , 0],
#     [20.5, 84.5, 0, 0, 0, 0 , 0 , 0, 0 , 0],
#     [3.1, 32.1, 84.2, 0, 0, 0 , 0 , 0, 0 , 0],
#     [0, 2.4, 39.5, 83.9, 0, 0 , 0 , 0, 0 , 0],
#     [0, 0, 0, 32.4, 85.1, 0 , 0 , 0, 0 , 0],
#     [0, 0, 0, 0, 38.8, 84.5, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 46.1, 84, 0, 0, 0],
#     [0, 0, 1.2, 0, 0, 0, 29.1, 83.2, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 29.2, 82, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 47, 82.5],
#
# ])
# 85, 52, 41, 31, 23.5, 20.55, 18.6, 12.2, 13
# matrix = np.array([
#     [86.4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [29.8, 81.1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [2.6, 27.8, 82.1, 0, 0, 0, 0, 0, 0, 0],
#     [4.9, 10.1, 48.4, 81.1, 0, 0, 0, 0, 0, 0],
#     [1.8, 2., 10.8, 39.8, 80.4, 0, 0, 0, 0, 0],
#     [2.5, 3.2, 9.9, 11.5, 41.3, 79.1, 0, 0, 0, 0],
#     [3.6, 3.4, 9.6, 9.1, 29.4, 56.5, 79.5, 0, 0, 0],
#     [6., 5.2, 10.8, 7.7, 20.1, 37.4, 41.6, 78.6, 0, 0],
#     [3.7, 3.4, 7., 7., 10., 21.4, 24.2, 49.2, 79.2, 0],
#     [7.7, 5.1, 11.7, 7.7, 9.7, 13.5, 20.3, 24.9, 44.7, 77.1],
#
# ])
# #  86.4, 55.5, 47. 5, 41, 36, 33, 24, 28.7, 25.5
# #35.3
#
#
# np.set_printoptions(formatter={'float': '{:0.1f}'.format})
# print(matrix)
