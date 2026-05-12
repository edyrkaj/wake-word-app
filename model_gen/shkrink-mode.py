from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "wake_word.onnx"
model_quant = "wake_word_quant.onnx"

quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QUInt8
)

print(f"Quantized model saved as {model_quant}")