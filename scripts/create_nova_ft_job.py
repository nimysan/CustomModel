#!/usr/bin/env python3
import boto3
import argparse
import logging
import time
import json
import os
import dotenv
from datetime import datetime
from pathlib import Path

# 解析命令行参数
def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='创建Amazon Bedrock Nova微调作业')
    
    parser.add_argument('--base-model-id', type=str, 
                        help='基础模型ID')
    
    parser.add_argument('--job-name', type=str, 
                        help='作业名称')
    
    parser.add_argument('--custom-model-name', type=str, 
                        help='自定义模型名称')
    
    parser.add_argument('--training-data-s3-uri', type=str, 
                        help='训练数据的S3 URI')
    
    parser.add_argument('--test-data-s3-uri', type=str, 
                        help='测试数据的S3 URI（可选）')
    
    parser.add_argument('--output-s3-uri', type=str, 
                        help='输出数据的S3 URI')
    
    parser.add_argument('--role-arn', type=str, 
                        help='IAM角色ARN')
    
    parser.add_argument('--region', type=str, 
                        help='AWS区域')
    
    parser.add_argument('--epoch-count', type=int, 
                        help='训练轮数')
    
    parser.add_argument('--batch-size', type=int, 
                        help='批处理大小')
    
    parser.add_argument('--learning-rate', type=float, 
                        help='学习率')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='打印作业配置而不创建作业')
    
    parser.add_argument('--skip-s3-check', action='store_true',
                        help='跳过检查S3中的训练数据')
    
    parser.add_argument('--config', type=str, default='../config.env',
                        help='配置文件路径')
    
    return parser.parse_args()

# 加载环境变量
def load_config(config_path):
    """加载环境变量配置。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    dotenv.load_dotenv(config_path)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # 获取必要的环境变量
    config = {
        'base_model_id': os.getenv('BASE_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
        'job_name': os.getenv('JOB_NAME', f"invoice-seller-extraction-{timestamp}"),
        'custom_model_name': os.getenv('CUSTOM_MODEL_NAME', f"invoice-seller-extraction-{timestamp}"),
        'training_data_s3_uri': f"s3://{os.getenv('S3_BUCKET')}/{os.getenv('S3_PREFIX_TRAINING')}/train_data.jsonl",
        'test_data_s3_uri': f"s3://{os.getenv('S3_BUCKET')}/{os.getenv('S3_PREFIX_TRAINING')}/test_data.jsonl",
        'output_s3_uri': f"s3://{os.getenv('S3_BUCKET')}/{os.getenv('S3_PREFIX_OUTPUT')}/",
        'role_arn': os.getenv('ROLE_ARN', f"arn:aws:iam::{os.getenv('AWS_ACCOUNT_ID')}:role/service-role/AmazonBedrockExecutionRoleForNova"),
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        'epoch_count': int(os.getenv('EPOCH_COUNT', '1')),
        'batch_size': int(os.getenv('BATCH_SIZE', '1')),
        'learning_rate': float(os.getenv('LEARNING_RATE', '0.0001')),
        'log_file': os.path.join('..', os.getenv('LOGS_DIR', 'output/logs'), 'nova_finetuning_job.log')
    }
    
    return config

def check_s3_file(s3_uri, region, required=True):
    """检查S3 URI是否包含指定的文件。"""
    try:
        # 解析S3 URI
        if not s3_uri.startswith('s3://'):
            logging.error(f"无效的S3 URI格式: {s3_uri}")
            return False if required else True
        
        parts = s3_uri[5:].split('/', 1)
        if len(parts) < 2:
            bucket = parts[0]
            key = ""
        else:
            bucket = parts[0]
            key = parts[1]
        
        # 创建S3客户端
        s3_client = boto3.client('s3', region_name=region)
        
        try:
            # 检查文件是否存在
            s3_client.head_object(Bucket=bucket, Key=key)
            logging.info(f"文件存在: {s3_uri}")
            return True
        except Exception as e:
            if required:
                logging.error(f"文件不存在: {s3_uri}")
                logging.error(f"错误: {e}")
                return False
            else:
                logging.info(f"可选文件不存在: {s3_uri}")
                return True
    
    except Exception as e:
        logging.error(f"检查S3文件时出错: {e}")
        return False if required else True

def create_fine_tuning_job(config):
    """使用boto3创建微调作业。"""
    try:
        # 检查训练数据是否存在于S3中
        if not config['skip_s3_check']:
            logging.info(f"检查训练数据: {config['training_data_s3_uri']}...")
            if not check_s3_file(config['training_data_s3_uri'], config['region'], required=True):
                logging.error("未找到训练数据。请先将训练数据上传到S3位置。")
                logging.error("您可以运行process_images_for_training.py脚本生成训练数据。")
                logging.error("或使用--skip-s3-check跳过此检查。")
                return None
            
            # 检查测试数据（如果提供）
            if config['test_data_s3_uri']:
                logging.info(f"检查测试数据: {config['test_data_s3_uri']}...")
                check_s3_file(config['test_data_s3_uri'], config['region'], required=False)
        
        # 创建Bedrock客户端
        bedrock_client = boto3.client('bedrock', region_name=config['region'])
        
        # 准备超参数
        hyperparameters = {
            "epochCount": str(config['epoch_count']),
            "batchSize": str(config['batch_size']),
            "learningRate": str(config['learning_rate'])
        }
        
        # 准备作业配置
        job_config = {
            "customizationType": "FINE_TUNING",
            "baseModelIdentifier": config['base_model_id'],
            "jobName": config['job_name'],
            "customModelName": config['custom_model_name'],
            "roleArn": config['role_arn'],
            "trainingDataConfig": {
                "s3Uri": config['training_data_s3_uri']
            },
            "outputDataConfig": {
                "s3Uri": config['output_s3_uri']
            },
            "hyperParameters": hyperparameters
        }
        
        # 如果提供了测试数据，添加到配置中
        if config['test_data_s3_uri'] and check_s3_file(config['test_data_s3_uri'], config['region'], required=False):
            job_config["validationDataConfig"] = {
                "validators": [
                    {
                        "s3Uri": config['test_data_s3_uri']
                    }
                ]
            }
        
        # 记录作业配置
        logging.info(f"作业配置: {json.dumps(job_config, indent=2, ensure_ascii=False)}")
        
        # 如果是模拟运行，只打印配置并退出
        if config['dry_run']:
            logging.info("模拟运行模式。未创建作业。")
            return None
        
        # 创建微调作业
        response = bedrock_client.create_model_customization_job(**job_config)
        
        # 记录响应
        job_id = response.get('jobArn', '').split('/')[-1]
        logging.info(f"微调作业创建成功。作业ID: {job_id}")
        logging.info(f"作业ARN: {response.get('jobArn')}")
        
        return response
    
    except Exception as e:
        logging.error(f"创建微调作业时出错: {e}")
        return None

def check_job_status(job_arn, region):
    """检查微调作业的状态。"""
    try:
        bedrock_client = boto3.client('bedrock', region_name=region)
        response = bedrock_client.get_model_customization_job(jobIdentifier=job_arn)
        status = response.get('status', 'UNKNOWN')
        logging.info(f"作业状态: {status}")
        return status
    except Exception as e:
        logging.error(f"检查作业状态时出错: {e}")
        return "ERROR"

def main():
    """创建微调作业的主函数。"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    if args.base_model_id:
        config['base_model_id'] = args.base_model_id
    if args.job_name:
        config['job_name'] = args.job_name
    if args.custom_model_name:
        config['custom_model_name'] = args.custom_model_name
    if args.training_data_s3_uri:
        config['training_data_s3_uri'] = args.training_data_s3_uri
    if args.test_data_s3_uri:
        config['test_data_s3_uri'] = args.test_data_s3_uri
    if args.output_s3_uri:
        config['output_s3_uri'] = args.output_s3_uri
    if args.role_arn:
        config['role_arn'] = args.role_arn
    if args.region:
        config['region'] = args.region
    if args.epoch_count:
        config['epoch_count'] = args.epoch_count
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    config['dry_run'] = args.dry_run
    config['skip_s3_check'] = args.skip_s3_check
    
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
    logging.info(f"- 基础模型ID: {config['base_model_id']}")
    logging.info(f"- 作业名称: {config['job_name']}")
    logging.info(f"- 自定义模型名称: {config['custom_model_name']}")
    logging.info(f"- 训练数据S3 URI: {config['training_data_s3_uri']}")
    if config['test_data_s3_uri']:
        logging.info(f"- 测试数据S3 URI: {config['test_data_s3_uri']}")
    logging.info(f"- 输出S3 URI: {config['output_s3_uri']}")
    logging.info(f"- 角色ARN: {config['role_arn']}")
    logging.info(f"- 区域: {config['region']}")
    logging.info(f"- 训练轮数: {config['epoch_count']}")
    logging.info(f"- 批处理大小: {config['batch_size']}")
    logging.info(f"- 学习率: {config['learning_rate']}")
    logging.info(f"- 模拟运行: {config['dry_run']}")
    logging.info(f"- 跳过S3检查: {config['skip_s3_check']}")
    
    # 创建微调作业
    response = create_fine_tuning_job(config)
    
    if response and not config['dry_run']:
        job_arn = response.get('jobArn')
        
        # 检查初始状态
        status = check_job_status(job_arn, config['region'])
        
        # 打印监控作业的说明
        logging.info("\n微调作业提交成功!")
        logging.info(f"作业ARN: {job_arn}")
        logging.info(f"初始状态: {status}")
        logging.info("\n要监控作业状态:")
        logging.info(f"  1. 使用AWS CLI: aws bedrock get-model-customization-job --job-identifier {job_arn} --region {config['region']}")
        logging.info(f"  2. 使用AWS控制台: https://{config['region']}.console.aws.amazon.com/bedrock/home?region={config['region']}#/modelcustomization")

if __name__ == "__main__":
    main()
