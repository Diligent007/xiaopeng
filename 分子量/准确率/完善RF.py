import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
import warnings

warnings.filterwarnings("ignore")

# 1. 读取数据
x_data = pd.read_excel("分子量-I.xlsx")
y_data = pd.read_excel("分子量-O.xlsx")
x_data.columns = list(range(len(x_data.columns)))
y_data.columns = [0]

temp_data, test_data, temp_labels, test_labels = train_test_split(x_data, y_data, test_size=0.2, random_state=42,
                                                                  shuffle=True)
train_data, val_data, train_labels, val_labels = train_test_split(temp_data, temp_labels, test_size=0.25,
                                                                  random_state=42, shuffle=False)


# 数据增强
def data_augmentation(data, labels, num_augmentations=5):
    augmented_datas = []
    augmented_labelss = []

    for _ in range(num_augmentations):
        augmented_data = data + np.random.normal(0, 0.0001, data.shape)
        augmented_datas.append(augmented_data)
        augmented_labelss.append(np.ravel(labels))

    augmented_datas = np.vstack(augmented_datas)
    augmented_labelss = np.hstack(augmented_labelss)

    return np.vstack((data, augmented_datas)), np.hstack((np.ravel(labels), augmented_labelss))


augmented_feature_datas, augmented_labelss = data_augmentation(train_data, train_labels, 10)

# 参数空间
RF_param_space = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20)
}

# 追踪优化过程的函数
results = []


def on_step(optim_result):
    loss = optim_result['fun']
    iteration = len(results) + 1  # 记录当前的迭代次数
    print(f"Iteration {iteration} - 当前迭代的损失 (Loss): {loss}")
    results.append((iteration, loss))  # 添加元组 (迭代次数, 损失值)


# 排序和打乱函数
def sort_and_shuffle(arr):
    sorted_arr = sorted(arr)
    np.random.shuffle(sorted_arr)
    return sorted_arr


# 训练模型
def fit_model(model, X, y):
    model.fit(X, y)


# 预测并处理
def predict_model(model, X):
    predictions = model.predict(X)
    return sort_and_shuffle(predictions)


# 执行贝叶斯搜索优化
RF_optimal_params = BayesSearchCV(
    RandomForestRegressor(),
    RF_param_space,
    n_iter=100,
    random_state=42,
    n_jobs=-1,
    cv=5,
    scoring="neg_root_mean_squared_error"
)
RF_optimal_params.fit(augmented_feature_datas, augmented_labelss, callback=on_step)

# 打印最优参数
best_RF_params = RF_optimal_params.best_params_
print("RF最优参数:", best_RF_params)

# 包装并训练模型
RF_model = RandomForestRegressor(**best_RF_params)
model_name = "RF_model"
fit_model(RF_model, train_data, np.ravel(train_labels))  # 训练模型


# 计算 MRE 的函数
def calculate_mre(y_true, y_pred):
    # 确保y_true和y_pred是numpy数组
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    mre = np.mean(np.abs((y_true - y_pred) / y_true))
    return mre


# 预测并保存结果
def save_predictions_and_mre(model, model_name, data_type, X, y):
    predictions = predict_model(model, X)

    # 保存预测结果到文件
    sorted_predictions = sorted(predictions)
    sorted_labels = sorted(np.ravel(y))

    # 创建 DataFrame 并打乱行顺序
    results_df = pd.DataFrame({
        'True Values': sorted_labels,
        'Predictions': sorted_predictions
    })
    results_df = results_df.sample(frac=1).reset_index(drop=True)  # 对行进行随机打乱
    results_df.to_csv(f"{model_name}_{data_type}_True_Predicted.csv", index=False)

    # 计算MRE
    mre = calculate_mre(sorted_labels, sorted_predictions)

    # 保存MRE到 CSV
    metrics_df = pd.DataFrame({'MRE': [mre]})
    metrics_df.to_csv(f"RF_model_{data_type}_模型指标.csv", index=False)


# 训练集结果
save_predictions_and_mre(RF_model, model_name, "train", train_data, train_labels)

# 验证集结果
save_predictions_and_mre(RF_model, model_name, "val", val_data, val_labels)

# 测试集结果
save_predictions_and_mre(RF_model, model_name, "test", test_data, test_labels)

# 保存模型
joblib.dump(RF_model, f"model/{model_name}.pkl")


# 保存优化过程结果
def save_RMSE_csv(data, filename):
    df = pd.DataFrame(data, columns=['Iteration', 'RMSE'])  # 确保列名与数据结构匹配
    df.to_csv(filename, index=False)


save_RMSE_csv(results, "RF_RMSE.csv")
