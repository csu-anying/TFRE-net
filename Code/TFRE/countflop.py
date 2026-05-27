# file: run_benchmark.py

import torch
import time
from thop import profile

# ===== 导入你的模型 =====
# from basemodel_judge_tree import PDF3
from basemodel_transformer_test import PDF3

class Wrapper(torch.nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x):
        labels = torch.zeros(x.size(0)).long().to(x.device)
        logit = self.model(x, self.num_classes, labels, "PTBXL", "CE",return_only_output=True)
        return logit


def benchmark(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    wrapper = Wrapper(model, 7).to(device)

    # ⚠️ 输入必须匹配你的模型
    x = torch.randn(1, 12, 128, 5000).to(device)

    # ===== Params =====
    params = sum(p.numel() for p in model.parameters())
    print("Params: %.3f M" % (params / 1e6))

    # ===== FLOPs =====
    flops, _ = profile(wrapper, inputs=(x,))
    print("FLOPs: %.3f G" % (flops / 1e9))

    # ===== Latency =====
    wrapper.eval()

    for _ in range(10):  # warmup
        wrapper(x)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(50):
        wrapper(x)

    torch.cuda.synchronize()
    end = time.time()

    print("Latency: %.3f ms" % ((end - start) / 50 * 1000))


if __name__ == "__main__":
    # ===== 模型初始化（必须和训练一致）=====
    dim_list = [5000] * 12
    hidden_dim = [32]
    num_class = 7

    model = PDF3(dim_list, hidden_dim, num_class, dropout=0.5)

    benchmark(model)