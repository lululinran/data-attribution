#导入库并加载数据

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# 创建图片输出文件夹
os.makedirs('images', exist_ok=True)

# 加载数据
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 数据探索可视化

# 散点图：酒精 vs 苹果酸
plt.figure(figsize=(6, 4))
sns.scatterplot(x=X['alcohol'], y=X['malic_acid'], hue=y, palette='Set2')
plt.title("Alcohol vs Malic Acid by Class")
plt.savefig('images/alcohol_malic_scatter.png')
plt.show()
plt.close()

# 类别分布图
plt.figure(figsize=(5, 3))
sns.countplot(x=y, palette='Set2')
plt.title("Class Distribution")
plt.xlabel("Wine Class")
plt.ylabel("Count")
plt.savefig('images/class_distribution.png')
plt.show()
plt.close()

#数据标准化

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA 降维分析

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 主成分方差贡献图
plt.figure(figsize=(6, 3))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.savefig('images/pca_variance_ratio.png')
plt.show()
plt.close()

# PCA 可视化
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1')
plt.title("PCA Projection (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig('images/pca_wine.png')
plt.show()
plt.close()

#t-SNE 降维分析

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# t-SNE 可视化
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='Set1')
plt.title("t-SNE Projection (2D)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.savefig('images/tsne_wine.png')
plt.show()
plt.close()

#KMeans 聚类分析

# PCA 上聚类
kmeans_pca = KMeans(n_clusters=3, random_state=42)
labels_pca = kmeans_pca.fit_predict(X_pca)
score_pca = silhouette_score(X_pca, labels_pca)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_pca, palette='Set3')
plt.title(f"KMeans Clustering on PCA (Score={score_pca:.2f})")
plt.savefig('images/pca_kmeans.png')
plt.show()
plt.close()

# t-SNE 上聚类
kmeans_tsne = KMeans(n_clusters=3, random_state=42)
labels_tsne = kmeans_tsne.fit_predict(X_tsne)
score_tsne = silhouette_score(X_tsne, labels_tsne)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_tsne, palette='Set3')
plt.title(f"KMeans Clustering on t-SNE (Score={score_tsne:.2f})")
plt.savefig('images/tsne_kmeans.png')
plt.show()
plt.close()

print(f"PCA Silhouette Score: {score_pca:.2f}")
print(f"t-SNE Silhouette Score: {score_tsne:.2f}")

#SVM 分类准确率

clf_pca = SVC(kernel='rbf', gamma='scale')
score_pca_svm = cross_val_score(clf_pca, X_pca, y, cv=5)
print(f"SVM Accuracy (PCA): {score_pca_svm.mean():.3f}")

clf_tsne = SVC(kernel='rbf', gamma='scale')
score_tsne_svm = cross_val_score(clf_tsne, X_tsne, y, cv=5)
print(f"SVM Accuracy (t-SNE): {score_tsne_svm.mean():.3f}")