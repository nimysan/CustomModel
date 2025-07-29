import numpy as np
import torch

# 示例1: 简单的线性量化 (32位浮点数 -> 8位整数)
def simple_quantize_float32_to_int8(data, scale=None):
    # 如果没有提供缩放因子，则自动计算
    if scale is None:
        # 找出数据范围
        data_max = np.max(np.abs(data))
        # 计算缩放因子 (127是int8的最大正值)
        scale = 127.0 / data_max if data_max > 0 else 1.0

    # 量化: 浮点数 -> 整数
    quantized_data = np.clip(np.round(data * scale), -128, 127).astype(np.int8)

    return quantized_data, scale

# 示例2: 反量化 (8位整数 -> 32位浮点数)
def dequantize_int8_to_float32(quantized_data, scale):
    # 反量化: 整数 -> 浮点数
    original_data = (quantized_data.astype(np.float32) / scale)
    return original_data

# 示例3: 使用PyTorch进行量化
def pytorch_quantization_example():
    # 创建一个32位浮点数张量
    x = torch.randn(5, 5)
    print("原始32位浮点数据:")
    print(x)

    # 量化为8位整数 (使用PyTorch的量化API)
    q_scale = 127.0 / torch.max(torch.abs(x))
    qx = torch.quantize_per_tensor(x, scale=q_scale, zero_point=0, dtype=torch.qint8)
    print("\n量化后的8位整数数据:")
    print(qx)

    # 反量化回32位浮点数
    x_dequantized = qx.dequantize()
    print("\n反量化后的32位浮点数据:")
    print(x_dequantized)

    # 计算量化误差
    error = torch.mean(torch.abs(x - x_dequantized))
    print(f"\n量化误差: {error.item()}")

# 演示代码
if __name__ == "__main__":
    # 创建一些32位浮点数据
    float32_data = np.random.randn(10).astype(np.float32) * 10
    print("原始32位浮点数据:")
    print(float32_data)

    # 量化为8位整数
    int8_data, scale_factor = simple_quantize_float32_to_int8(float32_data)
    print("\n量化后的8位整数数据:")
    print(int8_data)

    # 反量化回32位浮点数
    recovered_data = dequantize_int8_to_float32(int8_data, scale_factor)
    print("\n反量化后的32位浮点数据:")
    print(recovered_data)

    # 计算量化误差
    error = np.mean(np.abs(float32_data - recovered_data))
    print(f"\n量化误差: {error}")

    # 展示PyTorch的量化
    print("\n\nPyTorch量化示例:")
    pytorch_quantization_example()
