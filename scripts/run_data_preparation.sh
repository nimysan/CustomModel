#!/bin/bash

# 加载环境变量
CONFIG_FILE="../config.env"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "错误: 配置文件 $CONFIG_FILE 不存在"
  exit 1
fi

source "$CONFIG_FILE"

# 确保AWS凭证已配置
if ! aws sts get-caller-identity &> /dev/null; then
  echo "AWS凭证未配置。请先运行 'aws configure'。"
  exit 1
fi

# 安装必要的Python包
pip install boto3 pandas python-dotenv

# 设置环境
echo "设置环境..."
./setup_environment.sh

# 运行数据准备脚本
echo "开始数据准备..."
python3 process_images_for_training.py "$@"

# 检查脚本是否成功运行
if [ $? -eq 0 ]; then
  echo "数据准备成功完成。"
  
  # 检查JSONL文件是否存在并计算行数
  JSONL_FILE="../${BEDROCK_FT_DIR}/training_data.jsonl"
  if [ -f "$JSONL_FILE" ]; then
    LINE_COUNT=$(wc -l < "$JSONL_FILE")
    echo "创建了包含 $LINE_COUNT 个训练样本的JSONL文件。"
  else
    echo "警告: JSONL文件未找到。"
  fi
  
  echo ""
  echo "微调的后续步骤:"
  echo "1. 使用upload_data_to_s3.py脚本将JSONL文件上传到S3:"
  echo "   python3 upload_data_to_s3.py"
  echo ""
  echo "2. 使用AWS CLI创建微调作业:"
  echo "aws bedrock create-model-customization-job \\"
  echo "  --customization-type FINE_TUNING \\"
  echo "  --base-model-identifier anthropic.claude-3-sonnet-20240229-v1:0 \\"
  echo "  --job-name \"invoice-seller-extraction\" \\"
  echo "  --role-arn \"arn:aws:iam::${AWS_ACCOUNT_ID}:role/service-role/AmazonBedrockExecutionRoleForNova\" \\"
  echo "  --custom-model-name \"invoice-seller-extraction\" \\"
  echo "  --training-data-config \"s3Uri=s3://${S3_BUCKET}/${S3_PREFIX_TRAINING}/\" \\"
  echo "  --hyperparameters \"epochCount=3,batchSize=1,learningRate=0.0001\" \\"
  echo "  --output-data-config \"s3Uri=s3://${S3_BUCKET}/${S3_PREFIX_OUTPUT}/\""
else
  echo "数据准备失败。请检查日志了解详情。"
fi
