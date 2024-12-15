#import library
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score


df = pd.read_csv('FOOD-DATA-GROUP5.csv')
print(df.head())
print(df.info())
print(df.describe())

# Tăng chiều rộng và hiển thị đầy đủ các cột
pd.set_option('display.max_columns', None)  # Không giới hạn số lượng cột
pd.set_option('display.width', 1000)       # Tăng chiều rộng dòng hiển thị
pd.set_option('display.max_colwidth', None)  # Không giới hạn độ dài của cột văn bản

# Kiểm tra dữ liệu thiếu
print(df.isnull().sum())


# Tách các cột số cần chuẩn hóa (nếu không được chỉ định, tự động chọn tất cả các cột số)
numeric_columns = [
    'Caloric Value', 'Fat', 'Carbohydrates', 'Sugars', 'Protein',
    'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water',
    'Calcium', 'Iron', 'Magnesium', 'Manganese' , 'Selenium', 
    'Phosphorus', 'Potassium', 'Zinc' , 
]

epsilon = 1e-3  # Giá trị nhỏ để thay thế 0
df[numeric_columns] = df[numeric_columns].replace(0, epsilon) # Xử lý các giá trị bằng 0

# Chuẩn hóa bằng Min-Max Scaling
scaler = MinMaxScaler()
food_data= df[numeric_columns].copy()
food_data[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Hiển thị dữ liệu
print("Dữ liệu sau khi chuẩn hóa (Min-Max Scaling):")
print(food_data.head())

# Giả định các ngưỡng bệnh cho dữ liệu
# Quy đổi các ngưỡng cho dữ liệu chuẩn hóa (khoảng [0, 1])
food_data['Heart Disease Risk'] = ((food_data['Cholesterol'] > 0.66) & (food_data['Fat'] > 0.5)).astype(int) 
food_data['Diabetes Risk'] = ((food_data['Sugars'] > 0.5) & (food_data['Carbohydrates'] > 0.07)).astype(int) 
food_data['Cancer Risk'] = ((food_data['Fat'] > 0.67) & (food_data['Protein'] < 0.3)).astype(int)  
food_data['Gout Risk'] = ((food_data['Protein'] > 0.5)).astype(int)  
food_data['Hypertension Risk'] = ((food_data['Sodium'] > 0.04) & (food_data['Calcium'] < 0.02)).astype(int) 

# Chọn các cột dinh dưỡng và nguy cơ bệnh
nutritional_features = ['Caloric Value', 'Fat', 'Carbohydrates', 'Sugars', 'Protein',
                        'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water',
                        'Calcium', 'Iron', 'Magnesium', 'Manganese' , 'Selenium', 
                        'Phosphorus', 'Potassium', 'Zinc']
disease_columns = ['Heart Disease Risk', 'Diabetes Risk', 'Cancer Risk', 'Gout Risk', 'Hypertension Risk']

# Lọc dữ liệu
subset = food_data[nutritional_features + disease_columns]

# Nếu có NaN, có thể loại bỏ các dòng chứa NaN hoặc điền giá trị thiếu
subset_cleaned = subset.dropna()  # Loại bỏ các dòng chứa NaN

# Tính ma trận tương quan
correlation_matrix = subset_cleaned.corr().loc[nutritional_features, disease_columns]
print(correlation_matrix)

# Vẽ heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap: Nutritional Features vs Disease Risks')
plt.xlabel('Nguy cơ bệnh')
plt.ylabel('Yếu tố dinh dưỡng')
plt.show()

########## NGUY CƠ MẮC BỆNH TIM ##########

# Chọn mục tiêu và yếu tố dự đoán
X = food_data[nutritional_features]
y = food_data['Heart Disease Risk']  

# Chọn 5 yếu tố quan trọng nhất
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Xem các yếu tố quan trọng
important_features = selector.get_support(indices=True)
print("Các yếu tố quan trọng nhất ảnh hưởng tới bệnh tim:", [nutritional_features[i] for i in important_features])

# Xác định biến mục tiêu và biến dự đoán
target_Heart_Disease = 'Heart Disease Risk' # Biến mục tiêu (target_Heart_Disease): Một trong các cột nguy cơ bệnh
features_Heart_Disease = ['Fat', 'Caloric Value', 'Cholesterol', 'Carbohydrates', 'Water']  # Biến dự đoán Các cột chứa thông tin dinh dưỡng 

X_heart = food_data[features_Heart_Disease]
print("Tập kiểm thử X_heart : \n", X_heart)
y_heart = food_data[target_Heart_Disease]
print("Tập kiểm thử y_heart : \n", y_heart)


# Chia nhỏ mô hình cho tập
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.4, random_state=42)  
print("Kích thước tập huấn luyện:", X_train_heart.shape)
print("Kích thước tập kiểm thử:", X_test_heart.shape)

# Khởi tạo mô hình
decision_tree = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Đào tạo mô hình
decision_tree.fit(X_train_heart, y_train_heart)
random_forest.fit(X_train_heart, y_train_heart)

print("Mô hình Decision Trees và Random Forests đã được huấn luyện.")

# Dự đoán trên tập kiểm thử
y_pred_dt = decision_tree.predict(X_test_heart)
y_pred_rf = random_forest.predict(X_test_heart)

# Tính các chỉ số đánh giá (tránh lỗi bằng zero_division)
accuracy_dt = accuracy_score(y_test_heart, y_pred_dt)
sensitivity_dt = recall_score(y_test_heart, y_pred_dt, pos_label=1, zero_division=0)
specificity_dt = recall_score(y_test_heart, y_pred_dt, pos_label=0, zero_division=0)

accuracy_rf = accuracy_score(y_test_heart, y_pred_rf)
sensitivity_rf = recall_score(y_test_heart, y_pred_rf, pos_label=1, zero_division=0)
specificity_rf = recall_score(y_test_heart, y_pred_rf, pos_label=0, zero_division=0)

# In kết quả
print("Đánh giá mô hình Decision Trees:")
print(f"Độ chính xác: {accuracy_dt}")
print(f"Độ nhạy: {sensitivity_dt}")
print(f"Độ đặc hiệu: {specificity_dt}")

print("\nĐánh giá mô hình Random Forests:")
print(f"Độ chính xác: {accuracy_rf}")
print(f"Độ nhạy: {sensitivity_rf}")
print(f"Độ đặc hiệu: {specificity_rf}")

# Kiểm tra nhãn và chỉ định đầy đủ nhãn trong ma trận nhầm lẫn
unique_labels = sorted(set(y_test_heart) | set(y_pred_dt) | set(y_pred_rf))  # Tập hợp tất cả các nhãn

# Tính toán độ quan trọng của các yếu tố
# Tính toán độ quan trọng của các thuộc tính
importances = random_forest.feature_importances_  
# Sử dụng các tên cột từ tập dữ liệu huấn luyện
feature_importance = pd.DataFrame({'Feature': X_train_heart.columns, 'Importance': importances}) 
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Hiển thị kết quả dưới dạng bảng
print("Độ quan trọng của các yếu tố dinh dưỡng và hành vi:")
print(feature_importance)

# Vẽ biểu đồ độ quan trọng
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Độ Quan Trọng của Các Yếu Tố Dinh Dưỡng trong Random Forest')
plt.show()

########## NGUY CƠ MẮC BỆNH GOUT ##########


# Chọn mục tiêu và yếu tố dự đoán
X = food_data[nutritional_features]
y = food_data['Gout Risk']  # bệnh gout

# Chọn 5 yếu tố quan trọng nhất
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Xem các yếu tố quan trọng
important_features = selector.get_support(indices=True)
print("Các yếu tố quan trọng nhất ảnh hưởng tới bệnh Gout:", [nutritional_features[i] for i in important_features])

# Biểu đồ boxplot so sánh lượng protein theo nguy cơ bệnh gout
sns.boxplot(x=food_data['Gout Risk'], y=food_data['Protein'])
plt.title('So sánh lượng chất Cholesterol theo nguy cơ bệnh gout')
plt.xlabel('Nguy cơ Bệnh Gout')
plt.ylabel('Lượng Chất Protein')
plt.show()

# Biến mục tiêu (target_gout_Disease): Một trong các cột nguy cơ bệnh
# Biến dự đoán (features_gout_Disease): Các cột chứa thông tin dinh dưỡng 
# Xác định biến mục tiêu và biến dự đoán
target_Gout_Disease = 'Gout Risk'  
features_Gout_Disease = ['Protein', 'Iron', 'Magnesium', 'Phosphorus', 'Zinc']  

X_gout = food_data[features_Gout_Disease]
print("Tập kiểm thử X_gout : \n", X_gout)
y_gout = food_data[target_Gout_Disease]
print("Tập kiểm thử y_gout : \n", y_gout)

# Chia nhỏ mô hình cho tập
X_train_gout, X_test_gout, y_train_gout, y_test_gout = train_test_split(X_gout, y_gout, test_size=0.4, random_state=42)  
print("Kích thước tập huấn luyện:", X_train_gout.shape)
print("Kích thước tập kiểm thử:", X_test_gout.shape)

# Khởi tạo mô hình
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Đào tạo mô hình
decision_tree.fit(X_train_gout, y_train_gout)
random_forest.fit(X_train_gout, y_train_gout)

print("Mô hình Decision Trees và Random Forests đã được huấn luyện.")

# Dự đoán trên tập kiểm thử
y_pred_dt = decision_tree.predict(X_test_gout)
y_pred_rf = random_forest.predict(X_test_gout)

# Tính các chỉ số đánh giá (tránh lỗi bằng zero_division)
accuracy_dt = accuracy_score(y_test_gout, y_pred_dt)
sensitivity_dt = recall_score(y_test_gout, y_pred_dt, pos_label=1, zero_division=0)
specificity_dt = recall_score(y_test_gout, y_pred_dt, pos_label=0, zero_division=0)

accuracy_rf = accuracy_score(y_test_gout, y_pred_rf)
sensitivity_rf = recall_score(y_test_gout, y_pred_rf, pos_label=1, zero_division=0)
specificity_rf = recall_score(y_test_gout, y_pred_rf, pos_label=0, zero_division=0)

# In kết quả
print("Đánh giá mô hình Decision Trees:")
print(f"Độ chính xác: {accuracy_dt}")
print(f"Độ nhạy: {sensitivity_dt}")
print(f"Độ đặc hiệu: {specificity_dt}")

print("\nĐánh giá mô hình Random Forests:")
print(f"Độ chính xác: {accuracy_rf}")
print(f"Độ nhạy: {sensitivity_rf}")
print(f"Độ đặc hiệu: {specificity_rf}")

# Kiểm tra nhãn và chỉ định đầy đủ nhãn trong ma trận nhầm lẫn
unique_labels = sorted(set(y_test_gout) | set(y_pred_dt) | set(y_pred_rf))  # Tập hợp tất cả các nhãn


# Tính toán độ quan trọng của các yếu tố
# Tính toán độ quan trọng của các thuộc tính
importances = random_forest.feature_importances_  
# Sử dụng các tên cột từ tập dữ liệu huấn luyện
feature_importance = pd.DataFrame({'Feature': X_train_gout.columns, 'Importance': importances}) 
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Hiển thị kết quả dưới dạng bảng
print("Độ quan trọng của các yếu tố dinh dưỡng và hành vi:")
print(feature_importance)

# Vẽ biểu đồ độ quan trọng
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Độ Quan Trọng của Các Yếu Tố Dinh Dưỡng trong Random Forest')
plt.show()

##############################################
########## NGUY CƠ MẮC BỆNH BÉO PHÌ ##########
##############################################


# Chọn mục tiêu và yếu tố dự đoán
X = food_data[nutritional_features]
y = food_data['Diabetes Risk'] 

# Chọn 5 yếu tố quan trọng nhất
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Xem các yếu tố quan trọng
important_features = selector.get_support(indices=True)
print("Các yếu tố quan trọng nhất ảnh hưởng tới bệnh Diabetes:", [nutritional_features[i] for i in important_features])

# Biến mục tiêu (target_Diabetes_Disease): Một trong các cột nguy cơ bệnh
# Biến dự đoán (features_Diabetes_Disease): Các cột chứa thông tin dinh dưỡng 
# Xác định biến mục tiêu và biến dự đoán
target_Diabetes_Disease = 'Diabetes Risk'  
features_Diabetes_Disease = ['Caloric Value', 'Carbohydrates', 'Sugars', 'Water', 'Potassium']  

X_Diabetes = food_data[features_Diabetes_Disease]
print("Tập kiểm thử X_Diabetes : \n", X_Diabetes)
y_Diabetes = food_data[target_Diabetes_Disease]
print("Tập kiểm thử y_Diabetes : \n", y_Diabetes)


# Chia nhỏ mô hình cho tập
X_train_Diabetes, X_test_Diabetes, y_train_Diabetes, y_test_Diabetes = train_test_split(X_Diabetes, y_Diabetes, test_size=0.4, random_state=42)  
print("Kích thước tập huấn luyện:", X_train_Diabetes.shape)
print("Kích thước tập kiểm thử:", X_test_Diabetes.shape)

# Khởi tạo mô hình
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Đào tạo mô hình
decision_tree.fit(X_train_Diabetes, y_train_Diabetes)
random_forest.fit(X_train_Diabetes, y_train_Diabetes)

print("Mô hình Decision Trees và Random Forests đã được huấn luyện.")

# Dự đoán trên tập kiểm thử
y_pred_dt = decision_tree.predict(X_test_Diabetes)
y_pred_rf = random_forest.predict(X_test_Diabetes)

# Tính các chỉ số đánh giá (tránh lỗi bằng zero_division)
accuracy_dt = accuracy_score(y_test_Diabetes, y_pred_dt)
sensitivity_dt = recall_score(y_test_Diabetes, y_pred_dt, pos_label=1, zero_division=0)
specificity_dt = recall_score(y_test_Diabetes, y_pred_dt, pos_label=0, zero_division=0)

accuracy_rf = accuracy_score(y_test_Diabetes, y_pred_rf)
sensitivity_rf = recall_score(y_test_Diabetes, y_pred_rf, pos_label=1, zero_division=0)
specificity_rf = recall_score(y_test_Diabetes, y_pred_rf, pos_label=0, zero_division=0)

# In kết quả
print("Đánh giá mô hình Decision Trees:")
print(f"Độ chính xác: {accuracy_dt}")
print(f"Độ nhạy: {sensitivity_dt}")
print(f"Độ đặc hiệu: {specificity_dt}")

print("\nĐánh giá mô hình Random Forests:")
print(f"Độ chính xác: {accuracy_rf}")
print(f"Độ nhạy: {sensitivity_rf}")
print(f"Độ đặc hiệu: {specificity_rf}")

# Kiểm tra nhãn và chỉ định đầy đủ nhãn trong ma trận nhầm lẫn
unique_labels = sorted(set(y_test_Diabetes) | set(y_pred_dt) | set(y_pred_rf))  # Tập hợp tất cả các nhãn


# Tính toán độ quan trọng của các yếu tố
# Tính toán độ quan trọng của các thuộc tính
importances = random_forest.feature_importances_  
# Sử dụng các tên cột từ tập dữ liệu huấn luyện
feature_importance = pd.DataFrame({'Feature': X_train_Diabetes.columns, 'Importance': importances}) 
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Hiển thị kết quả dưới dạng bảng
print("Độ quan trọng của các yếu tố dinh dưỡng và hành vi:")
print(feature_importance)

# Vẽ biểu đồ độ quan trọng
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Độ Quan Trọng của Các Yếu Tố Dinh Dưỡng trong Random Forest')
plt.show()

##############################################
########## NGUY CƠ MẮC BỆNH UNG THƯ ##########
##############################################


# Chọn mục tiêu và yếu tố dự đoán
X = food_data[nutritional_features]
y = food_data['Cancer Risk']  # bệnh ung thư 

# Chọn 5 yếu tố quan trọng nhất
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Xem các yếu tố quan trọng
important_features = selector.get_support(indices=True)
print("Các yếu tố quan trọng nhất ảnh hưởng tới bệnh (UNG THƯ )Cancer:", [nutritional_features[i] for i in important_features])

# Biến mục tiêu : Một trong các cột nguy cơ bệnh
# Biến dự đoán : Các cột chứa thông tin dinh dưỡng 
# Xác định biến mục tiêu và biến dự đoán
target_Cancer_Disease = 'Cancer Risk'  
features_Cancer_Disease = ['Caloric Value', 'Fat', 'Carbohydrates', 'Cholesterol', 'Water']  

X_Cancer = food_data[features_Cancer_Disease]
print("Tập kiểm thử X_heart : \n", X_Cancer)
y_Cancer = food_data[target_Cancer_Disease]
print("Tập kiểm thử y_heart : \n", y_Cancer)


# Chia nhỏ mô hình cho tập
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(X_Cancer, y_Cancer, test_size=0.5, random_state=42)  
print("Kích thước tập huấn luyện:", X_train_cancer.shape)
print("Kích thước tập kiểm thử:", X_test_cancer.shape)

# Khởi tạo mô hình
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Đào tạo mô hình
decision_tree.fit(X_train_cancer, y_train_cancer)
random_forest.fit(X_train_cancer, y_train_cancer)

print("Mô hình Decision Trees và Random Forests đã được huấn luyện cho nguy cơ mắc bệnh ung thư.")

# Dự đoán trên tập kiểm thử
y_pred_dt = decision_tree.predict(X_test_cancer)
y_pred_rf = random_forest.predict(X_test_cancer)

# Tính các chỉ số đánh giá (tránh lỗi bằng zero_division)
accuracy_dt = accuracy_score(y_test_cancer, y_pred_dt)
sensitivity_dt = recall_score(y_test_cancer, y_pred_dt, pos_label=1, zero_division=0)
specificity_dt = recall_score(y_test_cancer, y_pred_dt, pos_label=0, zero_division=0)

accuracy_rf = accuracy_score(y_test_cancer, y_pred_rf)
sensitivity_rf = recall_score(y_test_cancer, y_pred_rf, pos_label=1, zero_division=0)
specificity_rf = recall_score(y_test_cancer, y_pred_rf, pos_label=0, zero_division=0)

# In kết quả
print("Đánh giá mô hình Decision Trees:")
print(f"Độ chính xác: {accuracy_dt}")
print(f"Độ nhạy: {sensitivity_dt}")
print(f"Độ đặc hiệu: {specificity_dt}")

print("\nĐánh giá mô hình Random Forests:")
print(f"Độ chính xác: {accuracy_rf}")
print(f"Độ nhạy: {sensitivity_rf}")
print(f"Độ đặc hiệu: {specificity_rf}")

# Kiểm tra nhãn và chỉ định đầy đủ nhãn trong ma trận nhầm lẫn
unique_labels = sorted(set(y_test_cancer) | set(y_pred_dt) | set(y_pred_rf))  # Tập hợp tất cả các nhãn


# Tính toán độ quan trọng của các yếu tố
# Tính toán độ quan trọng của các thuộc tính
importances = random_forest.feature_importances_  
# Sử dụng các tên cột từ tập dữ liệu huấn luyện
feature_importance = pd.DataFrame({'Feature': X_train_cancer.columns, 'Importance': importances}) 
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Hiển thị kết quả dưới dạng bảng
print("Độ quan trọng của các yếu tố dinh dưỡng và hành vi:")
print(feature_importance)

# Vẽ biểu đồ độ quan trọng
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Độ Quan Trọng của Các Yếu Tố Dinh Dưỡng trong Random Forest')
plt.show()

















































































































































