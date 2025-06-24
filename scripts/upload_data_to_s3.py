#!/usr/bin/env python3
import os
import boto3
import argparse
import logging
import glob
import dotenv
from pathlib import Path

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='上传训练数据到S3用于Nova微调')
    
    parser.add_argument('--input-dir', type=str, help='包含训练文件的目录')
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
        'input_dir': os.path.join('..', os.getenv('BEDROCK_FT_DIR')),
        's3_bucket': os.getenv('S3_BUCKET'),
        's3_prefix': os.getenv('S3_PREFIX_TRAINING'),
        'region': 'us-east-1',  # 默认区域
        'log_file': os.path.join('..', os.getenv('UPLOAD_DATA_LOG'))
    }
    
    return config

def upload_files_to_s3(input_dir, s3_bucket, s3_prefix, region, dry_run=False):
    """上传训练文件到S3。"""
    try:
        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            logging.error(f"输入目录不存在: {input_dir}")
            return False
        
        # 在输入目录中查找JSONL文件
        jsonl_file = os.path.join(input_dir, "training_data.jsonl")
        if not os.path.exists(jsonl_file):
            logging.error(f"JSONL文件未找到: {jsonl_file}")
            return False
        
        logging.info(f"找到要上传的JSONL文件: {jsonl_file}")
        
        # 创建S3客户端
        s3_client = boto3.client('s3', region_name=region)
        
        # 上传JSONL文件
        file_name = os.path.basename(jsonl_file)
        s3_key = f"{s3_prefix}/{file_name}"
        
        if dry_run:
            logging.info(f"[模拟运行] 将上传 {jsonl_file} 到 s3://{s3_bucket}/{s3_key}")
            uploaded = True
        else:
            try:
                s3_client.upload_file(jsonl_file, s3_bucket, s3_key)
                logging.info(f"已上传 {file_name} 到 s3://{s3_bucket}/{s3_key}")
                uploaded = True
            except Exception as e:
                logging.error(f"上传 {file_name} 时出错: {e}")
                uploaded = False
        
        # 验证上传
        if not dry_run and uploaded:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=s3_bucket,
                    Prefix=s3_prefix
                )
                
                if 'Contents' in response:
                    s3_files = [obj['Key'] for obj in response['Contents']]
                    logging.info(f"已验证S3位置中的文件: s3://{s3_bucket}/{s3_prefix}")
                    
                    # 打印用于微调作业的S3 URI
                    logging.info(f"\n训练数据已准备就绪，位于: s3://{s3_bucket}/{s3_prefix}")
                    logging.info("您现在可以使用以下命令创建微调作业:")
                    logging.info(f"aws bedrock create-model-customization-job --customization-type FINE_TUNING --base-model-identifier anthropic.claude-3-sonnet-20240229-v1:0 --training-data-config \"s3Uri=s3://{s3_bucket}/{s3_prefix}\"")
            except Exception as e:
                logging.error(f"验证上传时出错: {e}")
        
        return uploaded
    
    except Exception as e:
        logging.error(f"将文件上传到S3时出错: {e}")
        return False

def main():
    """上传训练数据的主函数。"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    input_dir = args.input_dir if args.input_dir else config['input_dir']
    s3_bucket = args.s3_bucket if args.s3_bucket else config['s3_bucket']
    s3_prefix = args.s3_prefix if args.s3_prefix else config['s3_prefix']
    region = args.region if args.region else config['region']
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['log_file']),
            logging.StreamHandler()
        ]
    )
    
    # 记录配置
    logging.info(f"输入目录: {input_dir}")
    logging.info(f"S3存储桶: {s3_bucket}")
    logging.info(f"S3前缀: {s3_prefix}")
    logging.info(f"区域: {region}")
    logging.info(f"模拟运行: {args.dry_run}")
    
    # 上传文件到S3
    success = upload_files_to_s3(
        input_dir,
        s3_bucket,
        s3_prefix,
        region,
        args.dry_run
    )
    
    if success:
        logging.info("上传成功完成")
    else:
        logging.error("上传失败")

if __name__ == "__main__":
    main()
