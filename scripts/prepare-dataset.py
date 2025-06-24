#!/usr/bin/env python3
"""
脚本用于识别发票图片中的销售方信息
使用 AWS Bedrock 的 Claude 模型通过 converse API 进行图像识别
使用 cross-region inference 方式调用模型
"""

import os
import boto3
import json
import csv
from pathlib import Path
import logging
from io import BytesIO
from PIL import Image

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 图片目录路径
IMAGE_DIR = Path("/Users/yexw/PycharmProjects/nova-fine-tunning/InvoiceDatasets/dataset/images/vat_train")
OUTPUT_FILE = Path("/Users/yexw/PycharmProjects/nova-fine-tunning/invoice_sellers.csv")

# 初始化 Bedrock 客户端 - 使用 cross-region inference
def get_bedrock_client():
    try:
        # 创建主区域的 Bedrock 客户端
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"  # 主区域
        )
        return bedrock_runtime
    except Exception as e:
        logger.error(f"创建 Bedrock 客户端失败: {e}")
        raise

# 读取图片并转换为 WEBP 格式的二进制数据
def process_image(image_path):
    try:
        # 使用 PIL 打开图片
        image = Image.open(image_path)
        
        # 将图片转换为 WEBP 格式并保存到 BytesIO 缓冲区
        buffer = BytesIO()
        image.save(buffer, format="WEBP", quality=90)
        image_data = buffer.getvalue()
        
        return image_data
    except Exception as e:
        logger.error(f"图片处理失败 {image_path}: {e}")
        return None

# 使用 Bedrock 的 Claude 模型识别图片中的销售方 - 使用 cross-region inference
def extract_seller_info(client, image_data):
    try:
        # 构建请求体 - 使用正确的参数格式和指定模型版本
        request_body = {
            "modelId": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # 使用指定的 Claude 3.7 Sonnet 模型版本
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "webp",
                                "source": {
                                    "bytes": image_data
                                }
                            }
                        },
                        {
                            "text": "这是一张发票图片。请识别并提取出销售方名称。只需要返回销售方名称，不要有其他文字。请确保提取的是销售方（开票方），而不是购买方（收票方）。"
                        }
                    ]
                }
            ]
        }
        
        # 调用 Bedrock 的 Claude 模型
        response = client.converse(**request_body)
        
        # 解析响应
        response_content = response.get('messages', [{}])[0].get('content', [{}])
        seller_name = ""
        
            # 获取模型输出
        model_output = response.get('output')["message"]["content"][0]["text"]
        
        # 获取性能指标
        metrics = response.get('metrics', {})
        latency_ms = metrics.get('latencyMs', 'N/A')
        usage = response.get('usage', {})
        
        # 构建输出文本
        output_text = (
            f"模型输出:\n{model_output}\n\n"
            f"性能指标:\n"
            f"延迟：{latency_ms}ms\n"
            f"Token使用统计:\n"
            f"输入tokens：{usage.get('inputTokens', 'N/A')}\n"
            f"输出tokens：{usage.get('outputTokens', 'N/A')}\n"
            f"总tokens：{usage.get('totalTokens', 'N/A')}"
        )
        logger.info("the output is: "+model_output)
        return model_output
        
        # return seller_name
    except Exception as e:
        logger.error(f"提取销售方信息失败: {e}")
        return f"提取失败: {str(e)}"

def main():
    try:
        # 检查图片目录是否存在
        if not IMAGE_DIR.exists() or not IMAGE_DIR.is_dir():
            logger.error(f"图片目录不存在: {IMAGE_DIR}")
            return
        
        # 获取 Bedrock 客户端
        bedrock_client = get_bedrock_client()
        
        # 准备输出 CSV 文件
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['图片名称', '销售方'])
            
            # 处理目录中的所有图片
            image_files = [f for f in IMAGE_DIR.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            total_images = len(image_files)
            
            logger.info(f"开始处理 {total_images} 张图片...")
            
            for i, image_path in enumerate(image_files, 1):
                try:
                    logger.info(f"处理图片 {i}/{total_images}: {image_path.name}")
                    
                    # 处理图片
                    image_data = process_image(image_path)
                    if not image_data:
                        logger.warning(f"跳过图片 {image_path.name} - 处理失败")
                        csv_writer.writerow([image_path.name, "处理失败"])
                        continue
                    
                    # 提取销售方信息
                    seller_name = extract_seller_info(bedrock_client, image_data)
                    
                    # 写入 CSV
                    csv_writer.writerow([image_path.name, seller_name])
                    logger.info(f"已提取: {image_path.name} -> {seller_name}")
                    
                except Exception as e:
                    logger.error(f"处理图片 {image_path.name} 时出错: {e}")
                    csv_writer.writerow([image_path.name, f"处理失败: {str(e)}"])
        
        logger.info(f"处理完成! 结果已保存到 {OUTPUT_FILE}")
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()
