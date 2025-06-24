#!/bin/bash

# 完整的数据处理流水线脚本
# 执行从LLM标注生成到JSONL上传的全过程

# 设置日志文件
LOG_FILE="../output/logs/complete_pipeline.log"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# 创建日志目录（如果不存在）
mkdir -p ../output/logs

# 日志函数
log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# 检查命令执行状态
check_status() {
  if [ $1 -eq 0 ]; then
    log "✅ $2 成功"
  else
    log "❌ $2 失败，退出代码: $1"
    exit $1
  fi
}

# 开始处理
log "=== 开始执行完整数据处理流水线 ==="
log "时间: $TIMESTAMP"

# 步骤1: 使用LLM生成CSV标注数据（如果需要）
if [ "$1" == "--generate-labels" ] || [ "$1" == "-g" ]; then
  log "步骤1: 使用LLM生成标注数据..."
  python3 generate_labels_with_llm.py --input-dir="../InvoiceDatasets/dataset/images/vat_train" --output-file="../data/invoice_sellers.csv"
  check_status $? "LLM标注生成"
else
  log "步骤1: 跳过LLM标注生成 (使用 --generate-labels 或 -g 参数执行此步骤)"
fi

# 步骤2: 处理图像并创建训练数据
log "步骤2: 处理图像并创建训练数据..."
python3 process_images_for_training.py
check_status $? "图像处理和训练数据创建"

# 步骤3: 验证训练数据格式
log "步骤3: 验证训练数据格式..."
python3 nova_ft_dataset_validator.py --input-file="../InvoiceDatasets/label-data-for-nova-custom-fine-tunning/output/training_data.jsonl"
check_status $? "训练数据验证"

# 步骤4: 上传JSONL到S3
log "步骤4: 上传JSONL到S3..."
python3 upload_data_to_s3.py
check_status $? "JSONL上传到S3"

# 完成
log "=== 数据处理流水线执行完成 ==="
log "您现在可以使用以下命令创建微调作业:"
log "aws bedrock create-model-customization-job \\"
log "  --customization-type FINE_TUNING \\"
log "  --base-model-identifier anthropic.claude-3-sonnet-20240229-v1:0 \\"
log "  --job-name \"invoice-seller-extraction\" \\"
log "  --role-arn \"arn:aws:iam::390468416359:role/service-role/AmazonBedrockExecutionRoleForNova\" \\"
log "  --custom-model-name \"invoice-seller-extraction\" \\"
log "  --training-data-config \"s3Uri=s3://aigcdemo.plaza.red/nova-fine-tunning/training-data/\" \\"
log "  --hyperparameters \"epochCount=3,batchSize=1,learningRate=0.0001\" \\"
log "  --output-data-config \"s3Uri=s3://aigcdemo.plaza.red/nova-fine-tunning/output/\""

# 显示处理结果摘要
JSONL_FILE="../InvoiceDatasets/label-data-for-nova-custom-fine-tunning/output/training_data.jsonl"
if [ -f "$JSONL_FILE" ]; then
  LINE_COUNT=$(wc -l < "$JSONL_FILE")
  log "创建了包含 $LINE_COUNT 个训练样本的JSONL文件"
fi

exit 0
