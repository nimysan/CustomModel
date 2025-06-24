#!/bin/bash

# 完整的数据处理流水线脚本
# 执行从LLM标注生成到JSONL上传和创建微调作业的全过程

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
./validate_jsonl.sh lite
check_status $? "训练数据验证"

# 步骤4: 上传JSONL到S3
log "步骤4: 上传JSONL到S3..."
python3 jsonl_to_s3.py $S3_BUCKET_ARG
check_status $? "JSONL上传到S3"

# 步骤5: 创建微调作业（可选）
if [ "$1" == "--create-job" ] || [ "$1" == "-c" ] || [ "$2" == "--create-job" ] || [ "$2" == "-c" ]; then
  log "步骤5: 创建微调作业..."
  python3 create_nova_ft_job.py $S3_BUCKET_ARG
  check_status $? "创建微调作业"
else
  log "步骤5: 跳过创建微调作业 (使用 --create-job 或 -c 参数执行此步骤)"
  log "您可以稍后使用以下命令创建微调作业:"
  log "python3 create_nova_ft_job.py $S3_BUCKET_ARG"
fi

# 完成
log "=== 数据处理流水线执行完成 ==="

# 显示处理结果摘要
TRAIN_JSONL="../${TRAIN_JSONL}"
TEST_JSONL="../${TEST_JSONL}"

if [ -f "$TRAIN_JSONL" ]; then
  TRAIN_COUNT=$(wc -l < "$TRAIN_JSONL")
  log "创建了包含 $TRAIN_COUNT 个训练样本的训练集JSONL文件"
fi

if [ -f "$TEST_JSONL" ]; then
  TEST_COUNT=$(wc -l < "$TEST_JSONL")
  log "创建了包含 $TEST_COUNT 个测试样本的测试集JSONL文件"
fi

log "所有处理结果保存在以下位置:"
log "- 训练集JSONL: $TRAIN_JSONL"
log "- 测试集JSONL: $TEST_JSONL"
log "- S3训练数据: s3://${S3_BUCKET}/${S3_PREFIX_TRAINING}/"
log "- S3图像数据: s3://${S3_BUCKET}/${S3_PREFIX_IMAGES}/"
log "- S3输出数据: s3://${S3_BUCKET}/${S3_PREFIX_OUTPUT}/"
log "- 日志文件: ../${LOGS_DIR}/"

exit 0
