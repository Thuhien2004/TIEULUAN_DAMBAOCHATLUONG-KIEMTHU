# TIEULUAN_DAMBAOCHATLUONG-KIEMTHU_code chương trình

# Ngôn ngữ: Python | Nền tảng: PyCharm & Anaconda local

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, completeness_score, silhouette_score
from sentence_transformers import SentenceTransformer



# BƯỚC 1: THU THẬP VÀ ĐỊNH DẠNG DỮ LIỆU (DATA COLLECTION)
 
``` def step_1_data_collection():
    print("[Bước 1] Đang nạp và định dạng dữ liệu kịch bản kiểm thử thô...")

    # Giả lập dữ liệu trích xuất từ hệ thống quản lý kiểm thử (Jira/TestRail)
    mock_test_cases = {
        'Test_ID': ['TC01', 'TC02', 'TC03', 'TC04', 'TC05', 'TC06', 'TC07', 'TC08', 'TC09', 'TC10'],
        'Description': [
            "Người dùng đăng nhập thất bại khi nhập sai mật khẩu",
            "Login failed do nhập sai password hệ thống",
            "Kiểm tra tính năng quên mật khẩu qua Email",
            "Thêm sản phẩm giày Adidas vào giỏ hàng thành công",
            "Bỏ mặt hàng giày vào cart của người dùng",
            "Xóa sản phẩm ra khỏi giỏ hàng trực tuyến",
            "Thanh toán hóa đơn bằng ví điện tử Momo",
            "Sử dụng ví Momo để hoàn tất thanh toán đơn hàng",
            "Thanh toán qua thẻ quốc tế Visa/Mastercard",
            "Kiểm tra hiển thị giao diện trang chủ khi tải lại"
        ]
    }
    # Định dạng dữ liệu thô về cấu trúc bảng (DataFrame)
    df_raw = pd.DataFrame(mock_test_cases)
    return df_raw
```

# BƯỚC 2: XÂY DỰNG GROUND TRUTH THÔNG QUA GÁN NHÃN (ANNOTATION)
```
def step_2_build_ground_truth(df):
    print("[Bước 2] Đang tích hợp dữ liệu Ground Truth từ chuyên gia QA...")

    # Mảng nhãn do hội đồng chuyên gia QA gán thủ công thông qua biểu quyết đa số
    # Các kịch bản trùng logic sẽ có chung mã số nhãn (Ví dụ: TC01 và TC02 đều là nhãn 0)
    expert_labels = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6]

    # Nhúng nhãn chuẩn vào cấu trúc dữ liệu để làm cơ sở đối chiếu (Sự thật nền)
    df['Ground_Truth'] = expert_labels
    return df

```

# BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU VÀ HUẤN LUYỆN AI (VECTORIZATION & CLUSTERING)

```
def step_3_ai_training_and_clustering(df):
    print("[Bước 3] Đang tiến hành tiền xử lý, trích xuất đặc trưng và phân cụm tự động...")

    # 3.1. Vectorization: Sử dụng mô hình Deep Learning đa ngôn ngữ để nhúng ngữ nghĩa câu lệnh
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    X_vectors = embedding_model.encode(df['Description'].tolist())

    # 3.2. Tìm số cụm tối ưu toán học bằng thuật toán phân tích Silhouette Score (Tránh lỗi mớm bài)
    best_k = 2
    best_score = -1
    for k in range(2, 9):
        test_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = test_kmeans.fit_predict(X_vectors)
        score = silhouette_score(X_vectors, labels)
        if score > best_score:
            best_score = score
            best_k = k
    print(f"   -> Kết quả phân tích toán học tự động xác định số nhóm tối ưu là: K = {best_k}")

    # 3.3. Clustering: Huấn luyện thuật toán K-Means chính thức với số cụm tối ưu đã tìm được
    kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['AI_Cluster'] = kmeans_model.fit_predict(X_vectors)

    return df, X_vectors, best_k

```

# BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH VÀ TRỰC QUAN HÓA (MODEL EVALUATION)

```
def step_4_model_evaluation(df, X_vectors, best_k):
    print("[Bước 4] Tiến hành đánh giá hiệu năng thuật toán dựa trên Ground Truth...")
    print("\n" + "=" * 20 + " BẢNG ĐỐI CHIẾU DỮ LIỆU THỰC TẾ " + "=" * 20)
    print(df[['Test_ID', 'Ground_Truth', 'AI_Cluster']])
    print("=" * 72)

    # 4.1. Tính toán các chỉ số chất lượng thuật toán học không giám sát
    ari_score = adjusted_rand_score(df['Ground_Truth'], df['AI_Cluster'])
    completeness = completeness_score(df['Ground_Truth'], df['AI_Cluster'])

    print(f"-> Chỉ số trùng khớp phân cụm (ARI Score)     : {ari_score * 100:.2f}%")
    print(f"-> Chỉ số độ bao phủ nhóm (Completeness Score): {completeness * 100:.2f}%")
    print("=" * 72 + "\n")

    # 4.2. Trực quan hóa không gian dữ liệu bằng giải thuật giảm chiều PCA
    print("-> Đang xuất biểu đồ đồ thị không gian ngữ nghĩa...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_vectors)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['AI_Cluster'], cmap='rainbow', s=120, edgecolors='black')

    # Đính nhãn mã ca kiểm thử (TC01 - TC10) lên mặt phẳng tọa độ
    for i, txt in enumerate(df['Test_ID']):
        plt.annotate(txt, (X_2d[i, 0], X_2d[i, 1]), size=9, xytext=(5, 5), textcoords='offset points')

    plt.title(f'Bản đồ phân cụm Test Cases tự động (AI xác định K={best_k} nhóm dựa trên Silhouette)')
    plt.xlabel('Trục toán học PCA 1')
    plt.ylabel('Trục toán học PCA 2')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Xuất file ảnh chất lượng cao phục vụ chèn tài liệu báo cáo
    plt.savefig('ket_qua_giai_phap_4_buoc.png', dpi=300)
    print("-> Đã lưu ảnh kết quả 'ket_qua_giai_phap_4_buoc.png' vào thư mục dự án.")
    plt.show()

if __name__ == "__main__":
    print("--- KHỞI CHẠY HỆ THỐNG MÔ PHỎNG KIỂM THỬ THÔNG MINH ---\n")

    # Thực hiện tuần tự quy trình logic 4 bước
    df_step1 = step_1_data_collection()
    df_step2 = step_2_build_ground_truth(df_step1)
    df_step3, vectors, optimized_k = step_3_ai_training_and_clustering(df_step2)
    step_4_model_evaluation(df_step3, vectors, optimized_k)

    print("--- QUY TRÌNH KẾT THÚC THÀNH CÔNG ---")

```
