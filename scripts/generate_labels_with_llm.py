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
import argparse
import dotenv
from pathlib import Path
import logging
from io import BytesIO
from PIL import Image

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='使用LLM生成发票销售方标注数据')
    
    parser.add_argument('--input-dir', type=str, help='包含发票图像的目录')
    parser.add_argument('--output-file', type=str, help='输出CSV文件路径')
    parser.add_argument('--model', type=str, default='us.anthropic.claude-3-7-sonnet-20250219-v1:0', help='要使用的LLM模型')
    parser.add_argument('--batch-size', type=int, default=10, help='每批处理的图像数量')
    parser.add_argument('--config', type=str, default='../config.env', help='配置文件路径')
    
    return parser.parse_args()

def load_config(config_path):
    """加载环境变量配置。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    dotenv.load_dotenv(config_path)
    
    # 获取必要的环境变量
    config = {
        'input_dir': os.path.join('..', os.getenv('TRAIN_IMAGES_DIR')),
        'output_file': os.path.join('..', os.getenv('LABEL_DATA_CSV')),
        'log_file': os.path.join('..', os.getenv('LOGS_DIR'), 'generate_labels.log')
    }
    
    return config

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    input_dir = args.input_dir if args.input_dir else config['input_dir']
    output_file = args.output_file if args.output_file else config['output_file']
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['log_file']),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"使用配置:")
    logger.info(f"- 输入目录: {input_dir}")
    logger.info(f"- 输出文件: {output_file}")
    logger.info(f"- 模型: {args.model}")
    logger.info(f"- 批处理大小: {args.batch_size}")
    
    # 获取图像文件列表
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(input_dir).glob(ext))
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 创建Bedrock客户端
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'  # 使用支持Claude的区域
    )
    
    # 准备CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 处理每个图像
        for i, image_path in enumerate(image_files):
            try:
                logger.info(f"处理图像 {i+1}/{len(image_files)}: {image_path.name}")
                
                # 读取图像
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # 调用Claude模型
                response = invoke_claude_with_image(bedrock_runtime, args.model, image_bytes)
                
                # 解析响应
                seller_name = parse_claude_response(response)
                
                # 写入CSV
                writer.writerow({
                    'image_name': image_path.name,
                    'label': seller_name
                })
                
                logger.info(f"提取的销售方: {seller_name}")
                
                # 每批次后保存
                if (i + 1) % args.batch_size == 0:
                    csvfile.flush()
                    logger.info(f"已处理 {i+1}/{len(image_files)} 个图像")
                
            except Exception as e:
                logger.error(f"处理图像 {image_path.name} 时出错: {e}")
                # # 记录错误但继续处理
                # writer.writerow({
                #     'image_name': image_path.name,
                #     'label': f"提取失败: {str(e)}"
                # })
    
    logger.info(f"处理完成。结果保存到 {output_file}")

def invoke_claude_with_image(client, model_id, image_bytes):
    """调用Claude模型处理图像。"""
    # 将图像编码为base64
    import base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # 构建请求
    request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "这是一张中国增值税发票。请识别并提取出销售方名称。只需要返回销售方名称，不要有其他文字。请确保提取的是销售方（开票方），而不是购买方（收票方）。"
                    }
                ]
            }
        ]
    }
    
    # 调用模型
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request)
    )
    
    # 解析响应
    response_body = json.loads(response['body'].read().decode('utf-8'))
    return response_body

def parse_claude_response(response):
    """从Claude响应中解析销售方名称。"""
    try:
        content = response['content'][0]['text']
        # 清理响应（删除可能的前缀和后缀）
        content = content.strip()
        return content
    except Exception as e:
        logging.error(f"解析响应时出错: {e}")
        return "提取失败"

if __name__ == "__main__":
    main()
