import re
from collections import defaultdict
import math


def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    folds = []
    current_fold = []

    for line in lines:
        if line.startswith('epoch: 0,'):
            if current_fold:  # 完成前一折的收集
                folds.append(current_fold)
            current_fold = [line]  # 开始新的一折
        elif current_fold:  # 当前折的数据收集
            current_fold.append(line)

    if current_fold:  # 添加最后一折
        folds.append(current_fold)

    return folds[:10]  # 确保只取前10折


def extract_epoch_data(fold_lines, target_epoch):
    epoch_block_size = 8  # 每个epoch数据块的行数
    target_index = target_epoch * epoch_block_size

    if target_index >= len(fold_lines):
        return None

    # 解析主指标行
    epoch_line = fold_lines[target_index]
    metrics = {}

    # 改进的解析方法：匹配所有Eval指标
    pattern = r'(Eval\s+[\w\s]+):\s*([\d.]+)'
    matches = re.findall(pattern, epoch_line)

    for key, value in matches:
        key = key.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue

    # 解析类别准确率
    class_accs = []
    for i in range(1, 8):  # 接下来的7行
        if target_index + i >= len(fold_lines):
            class_accs.append(0.0)
            continue

        class_line = fold_lines[target_index + i]
        if ':' in class_line:
            parts = class_line.split(':', 1)
            if len(parts) > 1:
                acc_str = parts[1].strip()
                try:
                    class_accs.append(float(acc_str))
                except ValueError:
                    class_accs.append(0.0)
            else:
                class_accs.append(0.0)
        else:
            class_accs.append(0.0)

    return metrics, class_accs


def calculate_stats(values):
    """计算平均值和标准差，返回格式化字符串"""
    n = len(values)
    if n == 0:
        return "0.00", "0.00"

    mean = sum(values) / n
    if n == 1:
        return f"{mean:.2f}", "0.00"

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_dev = math.sqrt(variance)

    return f"{mean:.2f}", f"{std_dev:.2f}"


def main(file_path):
    folds = parse_file(file_path)
    if not folds:
        print("未找到有效的折数据")
        return

    # 使用字典存储每个指标在所有折上的值
    metric_values = defaultdict(list)
    class_values = [[] for _ in range(7)]  # 7个类别的准确率值

    target_metrics = [
        'Eval Acc', 'Eval PR AUC', 'Eval F1', 'Eval Precision', 'Eval Recall',
        'Eval MACRO AUC', 'Eval MACRO ACC', 'Eval F2'
    ]

    valid_folds = 0

    # 收集所有折的数据
    for fold_idx, fold in enumerate(folds):
        # 计算该折的最大epoch数
        max_epoch = (len(fold) - 1) // 8
        target_epoch = max_epoch - 15

        if target_epoch < 0:
            print(f"第{fold_idx + 1}折的最大epoch{max_epoch}太小，无法计算max_epoch-15")
            continue

        data = extract_epoch_data(fold, target_epoch)
        if not data:
            print(f"警告: 第{fold_idx + 1}折未找到epoch {target_epoch}的数据")
            continue

        valid_folds += 1
        metrics, class_accs = data

        # 收集主要指标（转换为百分比）
        for key in target_metrics:
            if key in metrics:
                metric_values[key].append(metrics[key] * 100)
            else:
                metric_values[key].append(0.0)  # 若指标缺失，填充0

        # 收集类别准确率（转换为百分比）
        for i in range(7):
            if i < len(class_accs):
                class_values[i].append(class_accs[i] * 100)
            else:
                class_values[i].append(0.0)  # 若类别缺失，填充0

    if valid_folds == 0:
        print("没有有效数据可计算")
        return

    print(f"10折平均结果 (取每折的最大epoch-15, 有效折数: {valid_folds}):")
    # 计算并输出主要指标的平均值和标准差
    for key in target_metrics:
        values = metric_values[key]
        mean, std = calculate_stats(values)
        print(f"{key}: {mean} ± {std}")

    # 计算并输出类别准确率的平均值和标准差
    print("\n各类别平均准确率:")
    for i, values in enumerate(class_values):
        mean, std = calculate_stats(values)
        print(f"Class {i} Acc: {mean} ± {std}")


if __name__ == "__main__":
    file_path = r"C:\Users\25477\Desktop\记录\NB-net\包含模块消融PTBXL\tr_PTBXL_weights_change.txt"  # 替换为您的文件名
    main(file_path)