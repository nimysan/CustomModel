#!/usr/bin/env python3
"""
Script to visualize detailed training metrics with additional analysis
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import dotenv
from scipy import stats

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='生成详细训练指标图表')
    
    parser.add_argument('--metrics-file', type=str, help='指标CSV文件路径')
    parser.add_argument('--output-dir', type=str, help='保存生成图表的目录')
    parser.add_argument('--show', action='store_true', help='显示图表而不是保存')
    parser.add_argument('--include-outliers', action='store_true', help='在分析中包含异常值')
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

def visualize_detailed_metrics(metrics_file, output_dir=None, show=False, include_outliers=False):
    """生成详细训练指标图表。"""
    # 读取CSV文件
    df = pd.read_csv(metrics_file)
    
    # 创建图形
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 训练损失随时间变化
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(df['step_number'], df['training_loss'], marker='o', linestyle='-', color='blue')
    ax1.set_title('Training Loss vs Step Number')
    ax1.set_xlabel('Step Number')
    ax1.set_ylabel('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # 为每个epoch添加标记
    epochs = df['epoch_number'].unique()
    for epoch in epochs:
        epoch_start = df[df['epoch_number'] == epoch].iloc[0]['step_number']
        ax1.axvline(x=epoch_start, color='red', linestyle='--', alpha=0.3)
        ax1.text(epoch_start, ax1.get_ylim()[1]*0.9, f'Epoch {epoch}', rotation=90, alpha=0.7)
    
    # 2. 每个epoch的平均损失
    ax2 = fig.add_subplot(2, 2, 2)
    epoch_avg_loss = df.groupby('epoch_number')['training_loss'].mean()
    ax2.bar(epoch_avg_loss.index, epoch_avg_loss.values, color='skyblue')
    ax2.set_title('Average Loss per Epoch')
    ax2.set_xlabel('Epoch Number')
    ax2.set_ylabel('Average Loss')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 损失分布直方图
    ax3 = fig.add_subplot(2, 2, 3)
    
    # 处理异常值
    if not include_outliers:
        # 使用IQR方法识别异常值
        Q1 = df['training_loss'].quantile(0.25)
        Q3 = df['training_loss'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_df = df[(df['training_loss'] >= lower_bound) & (df['training_loss'] <= upper_bound)]
    else:
        filtered_df = df
    
    sns.histplot(filtered_df['training_loss'], kde=True, ax=ax3)
    ax3.set_title('Training Loss Distribution' + (' (Outliers Removed)' if not include_outliers else ''))
    ax3.set_xlabel('Training Loss')
    ax3.set_ylabel('Frequency')
    
    # 4. 损失变化率（每步相对于前一步的变化）
    ax4 = fig.add_subplot(2, 2, 4)
    df['loss_change'] = df['training_loss'].diff()
    ax4.plot(df['step_number'][1:], df['loss_change'][1:], marker='o', linestyle='-', color='green')
    ax4.set_title('Loss Change Rate')
    ax4.set_xlabel('Step Number')
    ax4.set_ylabel('Loss Change')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    # 添加总标题
    plt.suptitle('Detailed Training Metrics Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存或显示图表
    if show:
        plt.show()
    elif output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'detailed_training_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"详细指标图表已保存到: {output_path}")
    else:
        plt.show()

def main():
    """生成详细训练指标图表的主函数。"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    metrics_file = args.metrics_file if args.metrics_file else config['metrics_file']
    output_dir = args.output_dir if args.output_dir else config['output_dir']
    
    # 生成图表
    visualize_detailed_metrics(metrics_file, output_dir, args.show, args.include_outliers)

if __name__ == "__main__":
    main()
