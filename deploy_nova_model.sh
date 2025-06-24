#!/bin/bash

# 设置变量
MODEL_ARN="arn:aws:bedrock:us-east-1:390468416359:custom-model/amazon.nova-lite-v1:0:300k/4e15h395ng6o"
REGION="us-east-1"
THROUGHPUT_AMOUNT=1  # 购买的PT数量，可以根据需要调整
MODEL_NAME="invoice-seller-extraction"  # 部署后的模型名称
PROVISIONED_MODEL_NAME="invoice-seller-extraction-deployed"  # 预置模型的名称

# 步骤1: 购买预置吞吐量(PT)
echo "正在购买 $THROUGHPUT_AMOUNT 个预置吞吐量(PT)..."
aws bedrock purchase-provisioned-model-throughput \
  --model-id $MODEL_ARN \
  --provisioned-model-name $PROVISIONED_MODEL_NAME \
  --model-units $THROUGHPUT_AMOUNT \
  --commitment-duration "1y" \
  --region $REGION

# 等待PT购买完成
echo "等待PT购买完成..."
sleep 10

# 步骤2: 检查PT状态
echo "检查PT状态..."
aws bedrock get-provisioned-model-throughput \
  --provisioned-model-name $PROVISIONED_MODEL_NAME \
  --region $REGION

# 步骤3: 创建模型调用端点
echo "正在创建模型调用端点..."
aws bedrock create-provisioned-model-endpoint \
  --provisioned-model-name $PROVISIONED_MODEL_NAME \
  --region $REGION

# 等待端点创建完成
echo "等待端点创建完成，这可能需要几分钟..."
aws bedrock wait provisioned-model-endpoint-in-service \
  --provisioned-model-name $PROVISIONED_MODEL_NAME \
  --region $REGION

# 步骤4: 验证端点状态
echo "验证端点状态..."
aws bedrock get-provisioned-model-endpoint \
  --provisioned-model-name $PROVISIONED_MODEL_NAME \
  --region $REGION

echo "部署完成！您现在可以使用以下命令测试您的模型:"
echo "aws bedrock invoke-model --model-id $MODEL_ARN --body '{\"prompt\": \"测试提示\"}' --region $REGION output.json"
echo "或者使用预置端点:"
echo "aws bedrock invoke-provisioned-model --provisioned-model-name $PROVISIONED_MODEL_NAME --body '{\"prompt\": \"测试提示\"}' --region $REGION output.json"
