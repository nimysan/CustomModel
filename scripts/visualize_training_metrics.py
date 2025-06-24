#!/usr/bin/env python3
"""
Script to visualize training metrics from step_wise_training_metrics.csv
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dotenv

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='生成训练指标图表')
    
    parser.add_argument('--metrics-file', type=str, help='指标CSV文件路径')
    parser.add_argument('--output-dir', type=str, help='保存生成图表的目录')
    parser.add_argument('--show', action='store_true', help='显示图表而不是保存')
    parser.add_argument('--config', type=str, default='../config.env', help='配置文件路径')
    
    return parser.parse_args()

def load_config(config_path):
    """加载环境变量配置。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    dotenv.load_dotenv(config_path)
    
    # 获取必要的环境变量
    config = {
        'metrics_file': os.path.join('..', os.getenv('MODELS_DIR'), 'step_wise_training_metrics.csv'),
        'output_dir': os.path.join('..', os.getenv('DOCS_DIR'))
    }
    
    return config

def visualize_metrics(metrics_file, output_dir=None, show=False):
    """生成训练指标图表。"""
    # 读取CSV文件
    df = pd.read_csv(metrics_file)
    
    # 创建图形和坐标轴
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失与步骤数
    plt.plot(df['step_number'], df['training_loss'], marker='o', linestyle='-', color='blue', label='Training Loss')
    
    # 为每个epoch添加标记
    epochs = df['epoch_number'].unique()
    for epoch in epochs:
        epoch_start = df[df['epoch_number'] == epoch].iloc[0]['step_number']
        plt.axvline(x=epoch_start, color='red', linestyle='--', alpha=0.3)
        plt.text(epoch_start, plt.ylim()[1]*0.9, f'Epoch {epoch}', rotation=90, alpha=0.7)
    
    # 添加标题和标签
    plt.title('Training Loss vs Step Number')
    plt.xlabel('Step Number')
    plt.ylabel('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存或显示图表
    if show:
        plt.show()
    elif output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_loss_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    else:
        plt.show()

def main():
    """生成训练指标图表的主函数。"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    metrics_file = args.metrics_file if args.metrics_file else config['metrics_file']
    output_dir = args.output_dir if args.output_dir else config['output_dir']
    
    # 生成图表
    visualize_metrics(metrics_file, output_dir, args.show)

if __name__ == "__main__":
    main()
