#!/usr/bin/env python3
import os
import argparse
import subprocess
import logging
import glob
import dotenv
from pathlib import Path

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='验证训练数据集')
    
    parser.add_argument('--input-dir', type=str, help='包含训练数据的目录')
    parser.add_argument('--report-file', type=str, help='保存验证报告的路径')
    parser.add_argument('--fix', action='store_true', help='尝试修复数据集中的常见问题')
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
        'report_file': os.path.join('..', os.getenv('LOGS_DIR'), 'validation_report.txt'),
        'log_file': os.path.join('..', os.getenv('VALIDATION_LOG'))
    }
    
    return config

def validate_jsonl_file(jsonl_path, report_file, fix=False):
    """验证JSONL文件格式。"""
    try:
        # 使用nova_ft_dataset_validator.py验证
        cmd = ['python3', 'nova_ft_dataset_validator.py', '--input_file', jsonl_path, '--model_name', 'lite']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        with open(report_file, 'w') as f:
            f.write("=== 训练数据验证报告 ===\n\n")
            f.write(f"验证文件: {jsonl_path}\n\n")
            
            if result.returncode == 0:
                f.write("✅ 验证成功! 数据格式符合Nova微调要求。\n")
                f.write("\n详细信息:\n")
                f.write(result.stdout)
                logging.info(f"验证成功: {jsonl_path}")
                return True
            else:
                f.write("❌ 验证失败! 数据格式存在问题。\n")
                f.write("\n错误信息:\n")
                f.write(result.stderr)
                f.write("\n详细信息:\n")
                f.write(result.stdout)
                
                if fix:
                    f.write("\n\n尝试修复问题...\n")
                    # 这里可以添加修复逻辑
                    f.write("自动修复功能尚未实现。请手动修复问题。\n")
                
                logging.error(f"验证失败: {jsonl_path}")
                return False
    
    except Exception as e:
        logging.error(f"验证过程中出错: {e}")
        with open(report_file, 'w') as f:
            f.write("=== 训练数据验证报告 ===\n\n")
            f.write(f"验证文件: {jsonl_path}\n\n")
            f.write(f"❌ 验证过程中出错: {e}\n")
        return False

def main():
    """验证训练数据集的主函数。"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    input_dir = args.input_dir if args.input_dir else config['input_dir']
    report_file = args.report_file if args.report_file else config['report_file']
    
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
    logging.info(f"报告文件: {report_file}")
    logging.info(f"修复模式: {args.fix}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 查找JSONL文件
    jsonl_file = os.path.join(input_dir, "training_data.jsonl")
    if not os.path.exists(jsonl_file):
        logging.error(f"JSONL文件未找到: {jsonl_file}")
        return False
    
    # 验证JSONL文件
    success = validate_jsonl_file(jsonl_file, report_file, args.fix)
    
    if success:
        logging.info(f"验证成功完成。报告保存在: {report_file}")
    else:
        logging.error(f"验证失败。报告保存在: {report_file}")
    
    return success

if __name__ == "__main__":
    main()
