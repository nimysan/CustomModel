#!/bin/bash

# 验证JSONL文件脚本
# 分别验证训练集和测试集的JSONL文件

# 加载环境变量
CONFIG_FILE="../config.env"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "错误: 配置文件 $CONFIG_FILE 不存在"
  exit 1
fi

source "$CONFIG_FILE"

# 设置日志文件
LOG_FILE="../${VALIDATION_LOG}"
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
    return 0
  else
    log "❌ $2 失败，退出代码: $1"
    return 1
  fi
}

# 解析命令行参数
MODEL_NAME="lite"
if [ -n "$1" ]; then
  MODEL_NAME="$1"
fi

# 开始验证
log "=== 开始验证JSONL文件 ==="
log "时间: $TIMESTAMP"
log "使用模型: $MODEL_NAME"

# 设置JSONL文件路径
TRAIN_JSONL="../${BEDROCK_FT_DIR}/train_data.jsonl"
TEST_JSONL="../${BEDROCK_FT_DIR}/test_data.jsonl"

# 验证训练集JSONL
TRAIN_VALID=true
if [ -f "$TRAIN_JSONL" ]; then
  log "验证训练集JSONL文件: $TRAIN_JSONL"
  python3 nova_ft_dataset_validator.py -i "$TRAIN_JSONL" -m "$MODEL_NAME"
  
  if check_status $? "训练集JSONL验证"; then
    TRAIN_COUNT=$(wc -l < "$TRAIN_JSONL")
    log "训练集JSONL包含 $TRAIN_COUNT 个样本"
  else
    log "训练集JSONL验证失败，请检查文件格式"
    TRAIN_VALID=false
  fi
else
  log "训练集JSONL文件不存在: $TRAIN_JSONL"
  TRAIN_VALID=false
fi

# 验证测试集JSONL（如果存在）
if [ -f "$TEST_JSONL" ]; then
  log "验证测试集JSONL文件: $TEST_JSONL"
  python3 nova_ft_dataset_validator.py -i "$TEST_JSONL" -m "$MODEL_NAME"
  
  if check_status $? "测试集JSONL验证"; then
    TEST_COUNT=$(wc -l < "$TEST_JSONL")
    log "测试集JSONL包含 $TEST_COUNT 个样本"
  else
    log "测试集JSONL验证失败，请检查文件格式"
  fi
else
  log "测试集JSONL文件不存在: $TEST_JSONL"
fi

# 总结
log "=== JSONL验证完成 ==="

# 如果训练集验证失败，则整体验证失败
if [ "$TRAIN_VALID" = false ]; then
  log "验证失败: 训练集JSONL必须存在且格式有效"
  exit 1
fi

exit 0
