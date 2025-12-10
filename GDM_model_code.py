import sys
import os
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from EML import select_features_cv_importance_multi_model3
from EML import train_and_evaluate_model
from EML import evaluate_model
from EML import get_scaler
from EML import read_csv_to_dict
from EML import save_results_to_csv
from EML import split_and_save_data


# args
input_dir = sys.argv[1]
freq_the = float(sys.argv[2])
name = sys.argv[3]
clin = input_dir + sys.argv[4]
group_name = "group"
exp = input_dir + sys.argv[5]
gene_file = input_dir + sys.argv[6]
colname = sys.argv[7]
train_test_validation_file = clin
train_test_validation = pd.read_csv(train_test_validation_file, index_col=0, sep="\t")
train_test_validation_count = train_test_validation.groupby('label').size()
print(train_test_validation_count)
scaler_type = sys.argv[8]
test_size = float(sys.argv[9])
data_dict = read_csv_to_dict(gene_file)
feature_selection = list(set(data_dict[colname]))

output_dir = input_dir + name + "/"
# if not exists create output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

clin = pd.read_csv(clin, index_col=0, sep="\t")
mlnc_expression = pd.read_csv(exp, index_col=0, sep="\t")
mlnc_expression = mlnc_expression[mlnc_expression.columns.intersection(clin.index)]

usesamplename = list(mlnc_expression.columns.intersection(clin.index))
mlnc_expression_use = mlnc_expression[usesamplename]
clin_use = clin.loc[usesamplename]
detected_in_samples = np.sum(mlnc_expression_use > 0, axis=1)
min_samples = freq_the * mlnc_expression_use.shape[1]
print(min_samples)
selected_genes = detected_in_samples >= min_samples
mlnc_expression_use = mlnc_expression_use[selected_genes]
mlnc_expression_use = mlnc_expression_use.loc[mlnc_expression_use.index.intersection(feature_selection)]
print(mlnc_expression_use.shape)
mlnc_expression_use_filtered = mlnc_expression_use



top_n_list = [50]
print("top_n_list:", top_n_list)
freq_list = [0.8]
cv_num = 10
freq_list_real = [math.floor(freq * cv_num) for freq in freq_list]
print(freq_list_real)
index_name = ['name', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'TN', 'TP', 'FN', 'FP', 'Feature_num']




X_train_test = mlnc_expression_use_filtered.T[train_test_validation['label'] == 'train_test']
y_train_test = clin_use[group_name][train_test_validation['label'] == 'train_test']



# 在布尔索引后立即打印X_train_test的形状
print("Shape of X_train_test after boolean indexing:", X_train_test.shape)

# 检查是否有任何行被标记为'train_test'
print("Number of 'train_test' labels:", sum(train_test_validation['label'] == 'train_test'))

# 如果可能的话，打印前几行来检查数据是否如预期
print("First few rows of X_train_test:")
print(X_train_test.head())

# 同样对于y_train_test也做同样的检查
print("Shape of y_train_test:", y_train_test.shape)
print("First few values of y_train_test:")
print(y_train_test.head())



model_random_state_num = 42
model_lists = {
    'RF': RandomForestClassifier(n_estimators=100, random_state=model_random_state_num),
    'SVC': SVC(kernel='linear', C=0.01, random_state=model_random_state_num, probability=True, gamma='auto'),
    'LR': LogisticRegression(random_state=model_random_state_num),
}
sub_file_name = name + "_random_state_" + str(model_random_state_num)
print("use random_state:", model_random_state_num)
output_dir = input_dir + name + "/random_state_" + str(model_random_state_num) + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
X_train, X_test, y_train, y_test = split_and_save_data(X_train_test, y_train_test, test_size=test_size, random_state=model_random_state_num, shuffle=True, stratify=y_train_test, name=sub_file_name, outdir=output_dir)
scaler = get_scaler(scaler_type)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
if train_test_validation_count['validation'] == 0:
    X_validation = None
    y_validation = None
else:
    X_validation = mlnc_expression_use_filtered.T[train_test_validation['label'] == 'validation']
    y_validation = clin_use[group_name][train_test_validation['label'] == 'validation']
    X_validation_scaled = scaler.transform(X_validation)
for model_name, model in model_lists.items():
    model_detail_output_dir = output_dir + model_name + "_detail/"
    if not os.path.exists(model_detail_output_dir):
        os.makedirs(model_detail_output_dir)
    select_results,detail_df = select_features_cv_importance_multi_model3(X_train, y_train, scaler_type=scaler_type, model=model, cv=cv_num, freq_list=freq_list_real, top_n_list=top_n_list,save_detail=True)
    save_results_to_csv(select_results, output_dir + sub_file_name + "_" + model_name + "_cv_importance.csv")
    detail_df.to_csv(output_dir + sub_file_name + "_" + model_name + "_cv_importance_train_detail.csv",index=False)
    train_re_df_all = pd.DataFrame(index=index_name)
    test_re_df_all = pd.DataFrame(index=index_name)
    if X_validation is not None:
        validation_re_df_all = pd.DataFrame(index=index_name)
    else:
        validation_re_df_all = None
    for top_n in top_n_list:
        for freq in freq_list_real:
            sub_name = str(model_name) + "_" + str(top_n) + "_" + str(freq)
            selected_indices_cv_importance = select_results[(top_n, freq)]
            print(f"the selected indices by cv importance for {sub_name} are: {selected_indices_cv_importance}")
            if len(selected_indices_cv_importance) != 0:
                X_train_selected = X_train_scaled[:, [X_train.columns.get_loc(c) for c in selected_indices_cv_importance]]
                X_test_selected = X_test_scaled[:, [X_test.columns.get_loc(c) for c in selected_indices_cv_importance]]
                if X_validation is not None:
                    X_validation_selected = X_validation_scaled[:, [X_validation.columns.get_loc(c) for c in selected_indices_cv_importance]]
                else:
                    X_validation_selected = None
                train_model, train_re_df, train_stat_df = train_and_evaluate_model(pd.DataFrame(X_train_selected), y_train, model, name=sub_name, outdir=model_detail_output_dir)
                train_re_df_all = pd.concat([train_re_df_all, train_stat_df.set_index('Metric')], axis=1)
                test_re_df, test_stat_df = evaluate_model(pd.DataFrame(X_test_selected), y_test, train_model, name=sub_name, outdir=model_detail_output_dir,type='test')
                test_re_df_all = pd.concat([test_re_df_all, test_stat_df.set_index('Metric')], axis=1)
                if X_validation is not None:
                    validation_re_df, validation_stat_df = evaluate_model(pd.DataFrame(X_validation_selected), y_validation, train_model, name=sub_name, outdir=model_detail_output_dir,type='validation')
                    validation_re_df_all = pd.concat([validation_re_df_all, validation_stat_df.set_index('Metric')], axis=1)
    train_re_df_all = train_re_df_all.T
    test_re_df_all = test_re_df_all.T
    train_re_df_all.to_csv(output_dir + sub_file_name + "_" + model_name + "_train_statistics_all.csv",index=False)
    test_re_df_all.to_csv(output_dir + sub_file_name + "_" + model_name + "_test_statistics_all.csv",index=False)
    if X_validation is not None:
        validation_re_df_all = validation_re_df_all.T
        validation_re_df_all.to_csv(output_dir + sub_file_name + "_" + model_name + "_validation_statistics_all.csv",index=False)
