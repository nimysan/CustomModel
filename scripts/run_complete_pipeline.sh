#!/bin/bash

# 完整的数据处理流水线脚本
# 执行从LLM标注生成到JSONL上传的全过程

# 加载环境变量
CONFIG_FILE="../config.env"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "错误: 配置文件 $CONFIG_FILE 不存在"
  exit 1
fi

source "$CONFIG_FILE"

# 设置日志文件
LOG_FILE="../${PIPELINE_LOG}"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# 创建日志目录（如果不存在）
mkdir -p "../${LOGS_DIR}"

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

# 解析命令行参数
S3_BUCKET_ARG=""
if [ -n "$1" ] && [[ "$1" != "--"* ]]; then
  S3_BUCKET_ARG="--s3-bucket $1"
  shift
fi

# 开始处理
log "=== 开始执行完整数据处理流水线 ==="
log "时间: $TIMESTAMP"

# 设置环境
log "设置环境..."
./setup_environment.sh
check_status $? "环境设置"

# 步骤1: 使用LLM生成CSV标注数据（如果需要）
if [ "$1" == "--generate-labels" ] || [ "$1" == "-g" ]; then
  log "步骤1: 使用LLM生成训练集和测试集标注数据..."
  python3 generate_labels_with_llm.py $S3_BUCKET_ARG
  check_status $? "LLM标注生成"
else
  log "步骤1: 跳过LLM标注生成 (使用 --generate-labels 或 -g 参数执行此步骤)"
fi

# 步骤2: 处理图像并创建训练数据
log "步骤2: 处理训练集和测试集图像并创建训练数据..."
python3 process_images_for_training.py $S3_BUCKET_ARG
check_status $? "图像处理和训练数据创建"

# 步骤3: 验证训练数据格式
log "步骤3: 验证训练数据格式..."
python3 nova_ft_dataset_validator.py --input-file="../${BEDROCK_FT_DIR}/training_data.jsonl" --model_name="lite"
check_status $? "训练数据验证"

# 步骤4: 上传JSONL到S3
log "步骤4: 上传JSONL到S3..."
python3 upload_data_to_s3.py $S3_BUCKET_ARG
check_status $? "JSONL上传到S3"

# 完成
log "=== 数据处理流水线执行完成 ==="
log "您现在可以使用以下命令创建微调作业:"
log "aws bedrock create-model-customization-job \\"
log "  --customization-type FINE_TUNING \\"
log "  --base-model-identifier anthropic.claude-3-sonnet-20240229-v1:0 \\"
log "  --job-name \"invoice-seller-extraction\" \\"
log "  --role-arn \"arn:aws:iam::${AWS_ACCOUNT_ID}:role/service-role/AmazonBedrockExecutionRoleForNova\" \\"
log "  --custom-model-name \"invoice-seller-extraction\" \\"
log "  --training-data-config \"s3Uri=s3://${S3_BUCKET}/${S3_PREFIX_TRAINING}/\" \\"
log "  --hyperparameters \"epochCount=3,batchSize=1,learningRate=0.0001\" \\"
log "  --output-data-config \"s3Uri=s3://${S3_BUCKET}/${S3_PREFIX_OUTPUT}/\""

# 显示处理结果摘要
JSONL_FILE="../${BEDROCK_FT_DIR}/training_data.jsonl"
if [ -f "$JSONL_FILE" ]; then
  LINE_COUNT=$(wc -l < "$JSONL_FILE")
  log "创建了包含 $LINE_COUNT 个训练样本的JSONL文件"
fi

exit 0
