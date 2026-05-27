# benchmark_pdf3.py

import torch
import time
from thop import profile
from basemodel_transformer_test import PDF3
from checked_trans import prepare_trte_data
from datasets import *

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, return_only_output=True)


def benchmark(model, test_loader):
    device = torch.device("cuda")

    model = model.to(device)
    model.eval()
    wrapper = Wrapper(model).to(device)

    # ===== 用真实数据 =====
    batch, y4, y5, y7 = next(iter(test_loader))
    batch = [b.to(device) for b in batch]

    # ===== Params =====
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params/1e6:.3f} M")

    # ===== FLOPs（可能失败，正常）=====
    try:
        flops, _ = profile(wrapper, inputs=(batch,), verbose=False)
        print(f"FLOPs: {flops/1e9:.3f} G")
    except Exception as e:
        print("FLOPs skipped:", e)

    # ===== latency =====
    with torch.no_grad():
        for _ in range(10):
            wrapper(batch)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(50):
            wrapper(batch)

        torch.cuda.synchronize()
        end = time.time()

    print(f"Latency: {(end-start)/50*1000:.3f} ms")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    dim_list = [5000] * 12
    hidden_dim = [32]
    num_class = 7

    model = PDF3(dim_list, hidden_dim, num_class, dropout=0.5)

    # ⚠️ 加这一句（否则GPU没用）
    model = model.cuda()

    # ===== 构造 test_loader =====
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(
        "PTBXL", 7, 1, num_view=12
    )

    dataset_te = matDataset(
        data_test_list,
        torch.zeros(len(data_test_list[0])),
        torch.zeros(len(data_test_list[0])),
        torch.zeros(len(data_test_list[0]))
    )

    test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=1)

    benchmark(model, test_loader)