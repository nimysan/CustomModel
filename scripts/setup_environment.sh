#!/bin/bash

# 加载环境变量配置
CONFIG_FILE="../config.env"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 加载配置
source "$CONFIG_FILE"

# 创建必要的目录结构
mkdir -p "../${IMAGES_DIR}"
mkdir -p "../${LABEL_DATA_DIR}"
mkdir -p "../${BEDROCK_FT_DIR}"
mkdir -p "../${LOGS_DIR}"
mkdir -p "../${MODELS_DIR}"

echo "环境设置完成。已创建以下目录:"
echo "- 图片目录: ${IMAGES_DIR}"
echo "- 标签数据目录: ${LABEL_DATA_DIR}"
echo "- Bedrock微调数据目录: ${BEDROCK_FT_DIR}"
echo "- 日志目录: ${LOGS_DIR}"
echo "- 模型目录: ${MODELS_DIR}"
