# 完整数据处理流水线使用指南

本文档介绍如何使用 `run_complete_pipeline.sh` 脚本执行从LLM标注生成到JSONL上传的完整数据处理流程。

## 流程概述

完整的数据处理流水线包括以下步骤：

1. **LLM标注生成**（可选）：使用大型语言模型为发票图像生成销售方标注数据，并保存为CSV文件
2. **图像处理与训练数据创建**：读取CSV数据，上传图片到S3，并为每个图像创建训练JSON
3. **训练数据验证**：验证生成的训练数据格式是否符合Nova微调要求
4. **JSONL上传到S3**：将训练数据上传到S3存储桶，为微调做准备

## 使用方法

### 基本用法

```bash
cd scripts
./run_complete_pipeline.sh
```

这将执行步骤2-4，跳过LLM标注生成步骤（假设CSV文件已存在）。

### 包含LLM标注生成

```bash
cd scripts
./run_complete_pipeline.sh --generate-labels
```

或者使用简短形式：

```bash
cd scripts
./run_complete_pipeline.sh -g
```

这将执行完整流程，包括使用LLM生成标注数据。

## 输出

- 所有处理日志将保存在 `output/logs/complete_pipeline.log`
- 生成的CSV文件将保存在 `data/invoice_sellers.csv`
- 生成的训练JSONL文件将保存在 `InvoiceDatasets/label-data-for-nova-custom-fine-tunning/output/training_data.jsonl`
- 训练数据将上传到S3存储桶 `aigcdemo.plaza.red` 的 `nova-fine-tunning/training-data/` 前缀下

## 后续步骤

完成数据处理流水线后，您可以使用AWS CLI或控制台创建微调作业。脚本输出中包含了创建微调作业的AWS CLI命令示例。

## 故障排除

如果流水线执行过程中出现错误：

1. 检查 `output/logs/complete_pipeline.log` 日志文件，查找错误信息
2. 确保AWS凭证已正确配置（`aws configure`）
3. 确保所有依赖的Python包已安装（boto3, pandas等）
4. 检查S3存储桶权限是否正确
