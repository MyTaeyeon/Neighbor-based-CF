import pandas as pd
from sklearn.model_selection import train_test_split
from CF import CF
import numpy as np
import matplotlib.pyplot as plt

# Khởi tạo danh sách để lưu trữ giá trị RMSE cho mỗi lần chạy
rmse_user_list = []
rmse_item_list = []

def stratified_train_test_split(data, test_size, random_state):
    users = data['user_id'].unique()
    items = data['item_id'].unique()
    train_indices = []
    test_indices = []

    # Ensure each user appears in both training and testing set
    for user in users:
        user_data = data[data['user_id'] == user]
        if len(user_data) > 1:
            train_data, test_data = train_test_split(user_data, test_size=test_size, random_state=random_state)
            train_indices.extend(train_data.index)
            test_indices.extend(test_data.index)
        else:
            train_indices.extend(user_data.index)

    train_data = data.loc[train_indices]
    test_data = data.loc[test_indices]

    # Ensure all items are included
    train_items = train_data['item_id'].unique()
    test_items = test_data['item_id'].unique()

    missing_train_items = np.setdiff1d(items, train_items)
    missing_test_items = np.setdiff1d(items, test_items)

    for item in missing_train_items:
        item_data = data[data['item_id'] == item]
        if not item_data.empty:
            train_indices.extend(item_data.index[:1])

    for item in missing_test_items:
        item_data = data[data['item_id'] == item]
        if not item_data.empty:
            test_indices.extend(item_data.index[:1])

    train_data = data.loc[train_indices].drop_duplicates()
    test_data = data.loc[test_indices].drop_duplicates()

    return train_data, test_data

for i in range(1, 51):
    # Tải dữ liệu từ file u.data
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('./input/ml-100k/u.data', sep='\t', names=column_names)

    training_data, testing_data = stratified_train_test_split(data, test_size=0.5, random_state=i)

    training_data.iloc[:, :2] -= 1
    testing_data.iloc[:, :2] -= 1

    # User-based CF
    rs = CF(training_data.to_numpy(), k=30, uuCF=1)
    rs.fit()

    n_tests = testing_data.shape[0]
    SE = 0  # squared error
    for n in range(n_tests):
        pred = rs.pred(testing_data.iloc[n, 0], testing_data.iloc[n, 1], normalized=0)
        SE += (pred - testing_data.iloc[n, 2]) ** 2

    RMSE_user = np.sqrt(SE / n_tests)
    rmse_user_list.append(RMSE_user)

    # Item-based CF
    rs = CF(training_data.to_numpy(), k=30, uuCF=0)
    rs.fit()

    SE = 0  # squared error
    for n in range(n_tests):
        pred = rs.pred(testing_data.iloc[n, 0], testing_data.iloc[n, 1], normalized=0)
        SE += (pred - testing_data.iloc[n, 2]) ** 2

    RMSE_item = np.sqrt(SE / n_tests)
    rmse_item_list.append(RMSE_item)
    print(i, "completed!")

# Vẽ đồ thị
plt.figure(figsize=(12, 6))
plt.plot(range(1, 51), rmse_user_list, label='User-based CF', color='deepskyblue')
plt.plot(range(1, 51), rmse_item_list, label='Item-based CF', color='red')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('RMSE over 50 Iterations for User-based and Item-based CF')
plt.legend()

# Lưu đồ thị vào file PNG
plt.savefig('rmse_comparison.png')
print("Đồ thị đã được lưu vào file 'rmse_comparison.png'.")
