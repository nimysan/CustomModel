#!/usr/bin/env python3
import os
import csv
import json
import boto3
import argparse
from pathlib import Path
import logging
import dotenv

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='处理图像并创建训练数据')
    parser.add_argument('--s3-bucket', type=str, help='S3存储桶名称')
    parser.add_argument('--config', type=str, default='../config.env', help='配置文件路径')
    return parser.parse_args()

# 加载环境变量
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    dotenv.load_dotenv(config_path)
    
    # 获取必要的环境变量
    config = {
        'csv_path': os.path.join('..', os.getenv('INVOICE_SELLERS_CSV')),
        'images_dir': os.path.join('..', os.getenv('IMAGES_DIR')),
        's3_bucket': os.getenv('S3_BUCKET'),
        's3_prefix': os.getenv('S3_PREFIX_IMAGES'),
        'account_id': os.getenv('AWS_ACCOUNT_ID'),
        'output_dir': os.path.join('..', os.getenv('BEDROCK_FT_DIR')),
        'log_file': os.path.join('..', os.getenv('DATA_PREPARATION_LOG'))
    }
    
    return config

# 主函数
def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 如果命令行提供了S3存储桶，则覆盖配置
    if args.s3_bucket:
        config['s3_bucket'] = args.s3_bucket
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['log_file']),
            logging.StreamHandler()
        ]
    )
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 设置输出JSONL文件路径
    output_jsonl = os.path.join(config['output_dir'], "training_data.jsonl")
    
    logging.info(f"使用配置:")
    logging.info(f"- CSV文件: {config['csv_path']}")
    logging.info(f"- 图片目录: {config['images_dir']}")
    logging.info(f"- S3存储桶: {config['s3_bucket']}")
    logging.info(f"- S3前缀: {config['s3_prefix']}")
    logging.info(f"- 输出目录: {config['output_dir']}")
    logging.info(f"- 输出JSONL: {output_jsonl}")
    
    # 读取CSV数据
    csv_data = read_csv_data(config['csv_path'])
    if not csv_data:
        logging.error("CSV文件中没有有效数据。退出。")
        return
    
    # 处理每个条目
    successful_entries = 0
    failed_entries = 0
    skipped_entries = 0
    
    # 打开JSONL文件进行写入
    with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        for entry in csv_data:
            try:
                image_name = entry['图片名称']
                seller_name = entry['销售方']
                
                # 验证销售方名称
                if not seller_name or len(seller_name.strip()) == 0:
                    logging.warning(f"销售方名称为空: {image_name}。跳过。")
                    skipped_entries += 1
                    continue
                    
                # 检查图像是否存在
                image_path = os.path.join(config['images_dir'], image_name)
                if not os.path.exists(image_path):
                    logging.warning(f"图像不存在: {image_path}")
                    failed_entries += 1
                    continue
                
                # 上传图像到S3
                s3_uri = upload_image_to_s3(image_path, image_name, config)
                if not s3_uri:
                    failed_entries += 1
                    continue
                
                # 创建训练数据
                training_data = create_training_data(image_name, seller_name, s3_uri, config)
                if training_data:
                    # 将训练数据作为单行写入JSONL文件
                    jsonl_file.write(json.dumps(training_data, ensure_ascii=False) + '\n')
                    successful_entries += 1
                else:
                    failed_entries += 1
                    
            except Exception as e:
                logging.error(f"处理条目时出错 {entry}: {e}")
                failed_entries += 1
    
    logging.info(f"数据准备完成。")
    logging.info(f"成功: {successful_entries}, 失败: {failed_entries}, 跳过: {skipped_entries}")
    logging.info(f"总处理: {successful_entries + failed_entries + skipped_entries}")
    
    # 打印输出文件位置
    if successful_entries > 0:
        logging.info(f"训练JSONL文件可在以下位置获取: {output_jsonl}")
        logging.info("下一步: 使用upload_data_to_s3.py将此文件上传到S3")

def read_csv_data(csv_path):
    """读取包含图像名称和销售方信息的CSV文件。"""
    data = []
    filtered_count = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 过滤提取失败的条目
                seller_name = row.get('销售方', '')
                if '提取失败' in seller_name or 'ThrottlingException' in seller_name:
                    logging.warning(f"跳过提取失败的条目: {row['图片名称']}")
                    filtered_count += 1
                    continue
                data.append(row)
        
        logging.info(f"成功从CSV读取 {len(data)} 个有效条目 (过滤掉 {filtered_count} 个提取失败的条目)")
        return data
    except Exception as e:
        logging.error(f"读取CSV文件时出错: {e}")
        return []

def upload_image_to_s3(image_path, image_name, config):
    """将图像上传到S3并返回S3 URI。"""
    s3_client = boto3.client('s3')
    s3_key = f"{config['s3_prefix']}/{image_name}"
    
    try:
        s3_client.upload_file(image_path, config['s3_bucket'], s3_key)
        s3_uri = f"{config['s3_bucket']}/{s3_key}"
        logging.info(f"成功上传 {image_name} 到S3: {s3_uri}")
        return s3_uri
    except Exception as e:
        logging.error(f"上传 {image_name} 到S3时出错: {e}")
        return None

def validate_training_data(training_data):
    """验证训练数据对象，确保其符合Nova要求。"""
    try:
        # 检查schema版本
        if training_data.get('schemaVersion') != "bedrock-conversation-2024":
            logging.warning("训练数据中的schema版本无效")
            return False
        
        # 检查系统消息
        if not training_data.get('system') or not isinstance(training_data.get('system'), list) or len(training_data.get('system')) == 0:
            logging.warning("训练数据中的系统消息缺失或无效")
            return False
        
        # 检查消息
        if not training_data.get('messages') or not isinstance(training_data.get('messages'), list) or len(training_data.get('messages')) < 2:
            logging.warning("训练数据中的消息缺失或无效")
            return False
        
        # 检查用户消息
        user_message = training_data.get('messages')[0]
        if user_message.get('role') != 'user' or not user_message.get('content'):
            logging.warning("训练数据中的用户消息无效")
            return False
        
        # 检查用户消息中的图像
        user_content = user_message.get('content', [])
        has_image = False
        for content_item in user_content:
            if 'image' in content_item:
                has_image = True
                image_data = content_item.get('image', {})
                if not image_data.get('format') or not image_data.get('source', {}).get('s3Location', {}).get('uri'):
                    logging.warning("训练数据中的图像数据无效")
                    return False
        
        if not has_image:
            logging.warning("训练数据中的用户消息中未找到图像")
            return False
        
        # 检查助手消息
        assistant_message = training_data.get('messages')[1]
        if assistant_message.get('role') != 'assistant' or not assistant_message.get('content'):
            logging.warning("训练数据中的助手消息无效")
            return False
        
        # 检查助手响应
        assistant_content = assistant_message.get('content', [])
        if len(assistant_content) == 0 or not assistant_content[0].get('text'):
            logging.warning("训练数据中的助手响应无效")
            return False
        
        return True
    
    except Exception as e:
        logging.error(f"验证训练数据时出错: {e}")
        return False

def create_training_data(image_name, seller_name, s3_uri, config):
    """为Nova微调创建训练数据对象。"""
    # 获取文件扩展名
    file_format = image_name.split('.')[-1]
    
    training_data = {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{
            "text": "You are a smart assistant that answers questions respectfully"
        }],
        "messages": [{
                "role": "user",
                "content": [{
                        "text": "这是一张发票图片。请识别并提取出销售方名称。只需要返回销售方名称，不要有其他文字。请确保提取的是销售方（开票方），而不是购买方（收票方）。"
                    },
                    {
                        "image": {
                            "format": file_format,
                            "source": {
                                "s3Location": {
                                    "uri": s3_uri,
                                    "bucketOwner": config['account_id']
                                }
                            }
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{
                    "text": seller_name
                }]
            }
        ]
    }
    
    # 验证创建的训练数据
    if validate_training_data(training_data):
        logging.info(f"已创建并验证 {image_name} 的训练数据")
        return training_data
    else:
        logging.error(f"已创建但验证失败的训练数据: {image_name}")
        return None

if __name__ == "__main__":
    main()
