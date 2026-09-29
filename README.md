# Phân tích và Dự đoán Rời bỏ Khách hàng (Customer Churn Prediction)

Báo cáo Bài tập lớn môn: **Nền tảng Lập trình cho Phân tích và Trực quan Dữ liệu**  
- **Giảng viên hướng dẫn:** Lê Thành Sách  
- **Học kỳ:** 261 | **Năm học:** 2026-2027  

---

## 📌 Giới thiệu dự án

Bài toán đặt ra là xây dựng một hệ thống học máy nhằm dự đoán khả năng rời bỏ dịch vụ của khách hàng (Customer Churn) dựa trên các đặc trưng về thông tin cá nhân, lịch sử sử dụng dịch vụ và thanh toán. Việc dự đoán chính xác giúp doanh nghiệp đưa ra các chiến lược giữ chân khách hàng hiệu quả.

---

## 📊 Mô tả tập dữ liệu

Tập dữ liệu sử dụng đáp ứng đầy đủ các ràng buộc của đề bài:
- **Quy mô:** Bao gồm $N \ge 2000$ mẫu (cụ thể là **3500 dòng**) và $D \ge 10$ cột (cụ thể là **12 cột**).
- **Đặc tính:** Chứa giá trị thiếu (*missing values*), giá trị ngoại lai (*outliers*), cùng các biến thuộc kiểu dữ liệu số (*numerical*) và dạng phân loại (*categorical*).

---

## 🚀 Quy trình thực hiện

### 1. Khám phá dữ liệu (EDA - Exploratory Data Analysis)
- Thống kê mô tả cơ bản các thuộc tính số bằng `df.describe()`.
- Trực quan hóa phân phối của biến mục tiêu (`Churn`) và xây dựng ma trận tương quan Pearson giữa các biến số.

### 2. Tiền xử lý và Chuẩn bị dữ liệu
- **Xử lý giá trị thiếu (Missing Values):**
  - Cột số (`Age`, `MonthlyCharges`): Điền bằng giá trị trung vị (*Median*).
  - Cột phân loại (`PaymentMethod`): Điền bằng giá trị xuất hiện nhiều nhất (*Mode*).
- **Xử lý giá trị ngoại lai (Outliers):**
  - Sử dụng phương pháp khoảng tứ phân vị ($IQR = Q_3 - Q_1$).
  - Giới hạn biên dưới ($Q_1 - 1.5 \times IQR$) và biên trên ($Q_3 + 1.5 \times IQR$) áp dụng cho cột `TotalCharges`.
- **Mã hóa & Chuẩn hóa:**
  - **One-Hot Encoding** cho các biến phân loại (`Gender`, `PaymentMethod`, `ContractType`).
  - **StandardScaler** cho các biến số (`Age`, `Tenure`, `MonthlyCharges`, `TotalCharges`).
- **Phân chia tập dữ liệu:** Chia dữ liệu theo tỷ lệ **60% Train**, **20% Validation**, và **20% Test** (sử dụng phân tầng `stratify=y`).

### 3. Xây dựng và Huấn luyện mô hình
Sử dụng thư viện `Scikit-learn` để huấn luyện hai mô hình phân loại:
1. **Logistic Regression**
2. **Random Forest Classifier**

### 4. Đánh giá mô hình
Đánh giá hiệu suất mô hình trên tập Validation và Test dựa trên các chỉ số:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## 💻 Mã nguồn Python chính

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Tải và kiểm tra dữ liệu
sns.set_theme(style="whitegrid")
df = pd.read_csv('customer_churn_data.csv')

# 2. Tiền xử lý dữ liệu
# Xử lý Missing values
num_cols_with_na = ['Age', 'MonthlyCharges']
for col in num_cols_with_na:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols_with_na = ['PaymentMethod']
for col in cat_cols_with_na:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Xử lý Outliers cho 'TotalCharges'
Q1 = df['TotalCharges'].quantile(0.25)
Q3 = df['TotalCharges'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['TotalCharges'] = np.where(df['TotalCharges'] < lower_bound, lower_bound, df['TotalCharges'])
df['TotalCharges'] = np.where(df['TotalCharges'] > upper_bound, upper_bound, df['TotalCharges'])

# Mã hóa và Scaling
categorical_cols = ['Gender', 'PaymentMethod', 'ContractType']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

scaler = StandardScaler()
numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Chia tập dữ liệu (60 Train / 20 Val / 20 Test)
X = df.drop(columns=['CustomerID', 'Churn'])
y = df['Churn']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# 3. Huấn luyện mô hình
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Đánh giá mô hình
def evaluate_model(model, X_v, y_v, model_name="Model"):
    y_pred = model.predict(X_v)
    acc = accuracy_score(y_v, y_pred)
    print(f"--- ĐÁNH GIÁ MÔ HÌNH: {model_name} ---")
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_v, y_pred))

evaluate_model(log_reg, X_val, y_val, "Logistic Regression")
evaluate_model(rf_model, X_val, y_val, "Random Forest")
