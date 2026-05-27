import os
import wfdb
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Saliency
from basemodel_transformer_test import PDF3
# from basemodel_judge_tree import PDF3

from scipy.signal import butter, filtfilt

# --- 1. 高通滤波去基线漂移（0.5Hz） ---
def remove_baseline(ecg, fs=500):
    b, a = butter(1, 0.5/(fs/2), 'highpass')
    return filtfilt(b, a, ecg)

# --- 2. 幅度归一化（让 P/QRS/T 更清晰） ---
def normalize_amplitude(ecg):
    ecg = ecg - np.mean(ecg)
    return ecg / (np.max(np.abs(ecg)) + 1e-8)

# --- 3. 完整标准化流程 ---
def standardize_ecg(ecg):
    fs = 500  # PTB-XL 默认500Hz
    ecg_clean = np.zeros_like(ecg)

    for i in range(12):
        lead = ecg[:, i]
        lead = remove_baseline(lead, fs)
        lead = normalize_amplitude(lead)
        ecg_clean[:, i] = lead

    return ecg_clean

# ---------- 工具函数 ----------
def remove_extension(filename):
    return os.path.splitext(filename)[0]

def generate_heatmap_from_gradient(grad):
    # 取绝对值并归一化
    grad = grad.abs().cpu().numpy()
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return grad.T  # 转置成 [12, time]

def plot_12_leads_heatmap(ecg, heatmap, filename):
    import matplotlib as mpl
    from matplotlib import gridspec

    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.linewidth'] = 0.6

    # ---- 🎨 色差增强：gamma 校正 + 对比度拉伸 ----
    # heatmap 是 [12, time]
    gamma = 0.5   # gamma < 1 => 明亮部分增强
    heatmap = heatmap ** gamma

    # 再做一次增强归一化，把可视化差异拉大
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    leads = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(6, 2, hspace=0.35)

    for i in range(12):
        ax = fig.add_subplot(gs[i])

        # 画 ECG 曲线
        ax.plot(ecg[:, i], linewidth=0.9, color="black")

        # 绘制增强后的热力图
        ax.imshow(
            heatmap[i][np.newaxis, :],
            extent=[0, ecg.shape[0], ecg[:, i].min(), ecg[:, i].max()],
            cmap="plasma",
            alpha=0.65,  # <-- 透明度提高，让亮区更显眼
            aspect="auto",
        )

        ax.set_title(leads[i], fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 统一 colorbar（增强风格）
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap="plasma", norm=norm)
    cbar_ax.set_title("Saliency", fontsize=9)

    plt.suptitle(
        f"Saliency Heatmap (Enhanced Contrast)\nTFRE-Net",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    save_path = f"heatmap_{filename}.png"
    plt.savefig(save_path, dpi=350, bbox_inches="tight")
    plt.close()

    print(f"🔥 Enhanced-contrast academic heatmap saved: {save_path}")



# ---------- 核心可视化函数 ----------
def custom_forward_func(x, model):
    # model.forward 期望输入为 list of 12 leads
    x = x.permute(0, 2, 1)  # [B, L, 12] -> [B, 12, L]
    x_list = [x[:, i, :] for i in range(12)]
    _, logits = model(x_list, dataset='PTBXL')
    return torch.sigmoid(logits)

def readpre_with_heatmap(filename):

    filename = remove_extension(filename)
    data = wfdb.rdsamp(filename)
    ecg_data = data[0]
    ecg_data = standardize_ecg(ecg_data)
    # ---------- 强制使用 CPU ----------
    device = torch.device("cpu")

    # ---------- 加载模型 ----------
    dataset_list = [5000] * 12
    model = PDF3(dataset_list, [32], 7, dropout=0.5)
    model.to(device)

    # 强制把 state_dict 中的所有参数转换为 CPU tensor
    state_dict = torch.load("./NB-model.pth", map_location=torch.device("cpu"))# NB-net
    # state_dict = torch.load("./model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.to("cpu")

    # ---------- 确保模型内部不会再用 GPU ----------
    for name, param in model.named_parameters():
        assert param.device.type == "cpu", f"{name} 仍在 {param.device}"

    # ---------- 转换输入 ----------
    tensor_data = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0).to(device)
    tensor_data.requires_grad_()

    # ---------- 生成 Saliency 热力图 ----------
    saliency = Saliency(lambda x: custom_forward_func(x, model))
    gradient = saliency.attribute(tensor_data, target=0, abs=False)

    heatmap = generate_heatmap_from_gradient(gradient.squeeze(0))
    filename_base = os.path.basename(filename)
    plot_12_leads_heatmap(ecg_data, heatmap, filename_base)

    # ---------- 获取模型预测 ----------
    x_list = [tensor_data[0, :, i].unsqueeze(0) for i in range(12)]
    _, logits = model(x_list, dataset='PTBXL')
    pred = torch.argmax(torch.sigmoid(logits), dim=1).item()
    return pred



if __name__ == "__main__":
    filename = "./records500/01000/01063_hr"  # 示例路径（不含 .dat）
    predicted_class = readpre_with_heatmap(filename)
    print(f"Predicted class: {predicted_class}")
