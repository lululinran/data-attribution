import matplotlib.pyplot as plt

# 假设你前面已经得到了如下结果：
acc_pca = 0.966
acc_tsne = 0.966

# 构造数据
methods = ['PCA', 't-SNE']
accuracies = [acc_pca, acc_tsne]
colors = ['skyblue', 'lightgreen']

# 创建柱状图
plt.figure(figsize=(6, 5))
plt.bar(methods, accuracies, color=colors)
plt.ylim(0.9, 1.0)
plt.ylabel('SVM Accuracy')
plt.title('SVM Accuracy Comparison (PCA vs. t-SNE)')

# 添加文本标注
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.002, f"{acc:.3f}", ha='center', fontsize=10)

# 保存图像
plt.tight_layout()
plt.savefig("svm_accuracy_comparison.png")
plt.show()