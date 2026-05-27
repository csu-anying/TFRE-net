import torch
import time
import numpy as np
from fvcore.nn import FlopCountAnalysis

# from basemodel_cab import PDF3
from basemodel_transformer import PDF3  # CBAM MRM ECGtrans TGLLnet

# =========================
# wrapper（关键：兼容 list input）
# =========================
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, None, None, None, "PTBXL", inference_only=True)  # ⚠️ 按你run函数参数


# =========================
# benchmark
# =========================
def benchmark(model, device="cuda"):

    model = model.to(device)
    model.eval()

    wrapper = ModelWrapper(model).to(device).eval()

    # ======================
    # dummy input（12 leads）
    # ======================
    batch_size = 1
    seq_len = 5000

    x = [torch.randn(batch_size, seq_len).to(device) for _ in range(12)]

    # ======================
    # Params
    # ======================
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParams: {params / 1e6:.3f} M")

    # ======================
    # FLOPs（fvcore稳定版）
    # ======================
    try:
        flops = FlopCountAnalysis(wrapper, (x,))
        print(f"FLOPs: {flops.total() / 1e9:.3f} G")
    except Exception as e:
        print("FLOPs failed:", e)

    # ======================
    # Latency
    # ======================
    with torch.no_grad():

        # warmup
        for _ in range(20):
            _ = wrapper(x)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(100):
            _ = wrapper(x)

        torch.cuda.synchronize()
        end = time.time()

    latency = (end - start) / 100 * 1000
    print(f"Latency: {latency:.3f} ms")


# =========================
# main
# =========================
if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    dim_list = [5000] * 12
    hidden_dim = [48]  # 8 16MCA 48transformer 8MLRESNET 32CBAM None_MRM  # change
    num_class = 7

    model = PDF3(dim_list, hidden_dim, num_class, dropout=0.5)

    benchmark(model)