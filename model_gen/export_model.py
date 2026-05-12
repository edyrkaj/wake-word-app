import os
import torch
import torch.nn as nn
from onnxruntime.quantization import quantize_dynamic, QuantType  # pyright: ignore[reportMissingImports]

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.abspath(os.path.join(HERE, "..", "public"))
os.makedirs(PUBLIC_DIR, exist_ok=True)
MODEL_FP32 = os.path.join(HERE, "model.onnx")
MODEL_QUANT = os.path.join(PUBLIC_DIR, "model_tiny.onnx")

# 1. Architecture: DS-CNN (Depthwise Separable CNN)
class WakeWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, groups=16), # Depthwise
            nn.Conv2d(16, 32, kernel_size=1),            # Pointwise
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, 2) # [Silence/Noise, WakeWord]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=1)

# 2. Export & Quantize
model = WakeWordModel().eval()
dummy_input = torch.randn(1, 1, 40, 99) # 40 Mel bins x 99 frames (~1s)

torch.onnx.export(
    model,
    dummy_input,
    MODEL_FP32,
    input_names=['input'],
    output_names=['output'],
    dynamo=False,
    opset_version=17,
)

quantize_dynamic(MODEL_FP32, MODEL_QUANT, weight_type=QuantType.QUInt8)
print(f"Tiny model generated: {MODEL_QUANT}")