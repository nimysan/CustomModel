#!/usr/bin/env python3
"""
将JSONL文件上传到S3的适当目录
"""

import os
import boto3
import argparse
import logging
import dotenv
from pathlib import Path

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='将JSONL文件上传到S3的适当目录')
    
    parser.add_argument('--train-jsonl', type=str, help='训练集JSONL文件路径')
    parser.add_argument('--test-jsonl', type=str, help='测试集JSONL文件路径')
    parser.add_argument('--s3-bucket', type=str, help='S3存储桶名称')
    parser.add_argument('--s3-prefix', type=str, help='S3前缀（文件夹路径）')
    parser.add_argument('--region', type=str, help='AWS区域')
    parser.add_argument('--dry-run', action='store_true', help='打印要上传的文件而不实际上传')
    parser.add_argument('--config', type=str, default='../config.env', help='配置文件路径')
    
    return parser.parse_args()

def load_config(config_path):
    """加载环境变量配置。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    dotenv.load_dotenv(config_path)
    
    # 获取必要的环境变量
    config = {
        'train_jsonl': os.path.join('..', os.getenv('TRAIN_JSONL', 'data/bedrock-ft/train_data.jsonl')),
        'test_jsonl': os.path.join('..', os.getenv('TEST_JSONL', 'data/bedrock-ft/test_data.jsonl')),
        's3_bucket': os.getenv('S3_BUCKET', 'aigcdemo.plaza.red'),
        's3_prefix': os.getenv('S3_PREFIX_TRAINING', 'nova-ft/training/data'),
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        'log_file': os.path.join('..', os.getenv('LOGS_DIR', 'output/logs'), 'jsonl_to_s3.log')
    }
    
    return config

def upload_file_to_s3(file_path, s3_bucket, s3_key, region, dry_run=False):
    """将文件上传到S3。"""
    if not os.path.exists(file_path):
        logging.error(f"文件不存在: {file_path}")
        return False
    
    if dry_run:
        logging.info(f"[模拟运行] 将上传 {file_path} 到 s3://{s3_bucket}/{s3_key}")
        return True
    
    try:
        s3_client = boto3.client('s3', region_name=region)
        s3_client.upload_file(file_path, s3_bucket, s3_key)
        logging.info(f"成功上传 {file_path} 到 s3://{s3_bucket}/{s3_key}")
        return True
    except Exception as e:
        logging.error(f"上传 {file_path} 到 s3://{s3_bucket}/{s3_key} 时出错: {e}")
        return False

def main():
    """将JSONL文件上传到S3的主函数。"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    train_jsonl = args.train_jsonl if args.train_jsonl else config['train_jsonl']
    test_jsonl = args.test_jsonl if args.test_jsonl else config['test_jsonl']
    s3_bucket = args.s3_bucket if args.s3_bucket else config['s3_bucket']
    s3_prefix = args.s3_prefix if args.s3_prefix else config['s3_prefix']
    region = args.region if args.region else config['region']
    
    # 配置日志
    os.makedirs(os.path.dirname(config['log_file']), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['log_file']),
            logging.StreamHandler()
        ]
    )
    
    # 记录配置
    logging.info(f"使用配置:")
    logging.info(f"- 训练集JSONL: {train_jsonl}")
    logging.info(f"- 测试集JSONL: {test_jsonl}")
    logging.info(f"- S3存储桶: {s3_bucket}")
    logging.info(f"- S3前缀: {s3_prefix}")
    logging.info(f"- 区域: {region}")
    logging.info(f"- 模拟运行: {args.dry_run}")
    
    # 上传训练集JSONL
    train_success = False
    if os.path.exists(train_jsonl):
        train_filename = os.path.basename(train_jsonl)
        train_s3_key = f"{s3_prefix}/{train_filename}"
        train_success = upload_file_to_s3(train_jsonl, s3_bucket, train_s3_key, region, args.dry_run)
    else:
        logging.error(f"训练集JSONL文件不存在: {train_jsonl}")
    
    # 上传测试集JSONL（如果存在）
    test_success = False
    if os.path.exists(test_jsonl):
        test_filename = os.path.basename(test_jsonl)
        test_s3_key = f"{s3_prefix}/{test_filename}"
        test_success = upload_file_to_s3(test_jsonl, s3_bucket, test_s3_key, region, args.dry_run)
    else:
        logging.info(f"测试集JSONL文件不存在: {test_jsonl}")
    
    # 验证上传
    if not args.dry_run:
        try:
            s3_client = boto3.client('s3', region_name=region)
            response = s3_client.list_objects_v2(
                Bucket=s3_bucket,
                Prefix=s3_prefix
            )
            
            if 'Contents' in response:
                s3_files = [obj['Key'] for obj in response['Contents']]
                logging.info(f"S3位置中的文件: s3://{s3_bucket}/{s3_prefix}")
                for i, file_key in enumerate(s3_files):
                    logging.info(f"  {i+1}. {file_key}")
                
                # 打印用于微调作业的S3 URI
                logging.info(f"\n训练数据已准备就绪，位于: s3://{s3_bucket}/{s3_prefix}")
                logging.info("您现在可以使用以下命令创建微调作业:")
                logging.info(f"python3 create_nova_ft_job.py --training-data-s3-uri s3://{s3_bucket}/{s3_prefix}/train_data.jsonl")
                if test_success:
                    logging.info(f"--test-data-s3-uri s3://{s3_bucket}/{s3_prefix}/test_data.jsonl")
        except Exception as e:
            logging.error(f"验证上传时出错: {e}")
    
    # 总结
    if train_success:
        logging.info("训练集JSONL上传成功")
    else:
        logging.error("训练集JSONL上传失败")
    
    if os.path.exists(test_jsonl):
        if test_success:
            logging.info("测试集JSONL上传成功")
        else:
            logging.error("测试集JSONL上传失败")

if __name__ == "__main__":
    main()
