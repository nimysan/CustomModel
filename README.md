# Amazon Bedrock Nova Fine-tuning for Invoice Seller Extraction

This project prepares training data for fine-tuning Amazon Bedrock Nova models to extract seller information from Chinese VAT invoices.

## Project Structure

```
nova-fine-tunning/
│
├── README.md                           # 项目说明文档
│
├── scripts/                            # 脚本文件目录
│   ├── process_images_for_training.py  # 处理图像和创建训练数据的脚本
│   ├── upload_data_to_s3.py            # 上传训练JSON文件到S3的脚本
│   ├── generate_labels_with_llm.py     # 使用LLM生成标注数据的脚本
│   ├── visualize_training_metrics.py   # 生成训练指标图表的脚本
│   ├── visualize_detailed_metrics.py   # 生成详细训练指标图表的脚本
│   ├── nova_ft_dataset_validator.py    # 验证训练数据格式的脚本
│   ├── validate_training_dataset.py    # 验证训练数据集的脚本
│   ├── run_data_preparation.sh         # 运行数据准备过程的Shell脚本
│   ├── run_complete_pipeline.sh        # 执行完整数据处理流水线的Shell脚本
│   └── setup_environment.sh            # 设置环境和目录结构的脚本
│
├── data/                               # 数据目录
│   ├── images/                         # 发票图像目录
│   ├── label_data/                     # 标注数据目录
│   │   └── invoice_sellers.csv         # 包含图像名称和销售方信息的CSV文件
│   └── bedrock-ft/                     # Bedrock微调数据目录
│       └── training_data.jsonl         # 训练JSONL文件
│
├── InvoiceDatasets/                    # 从GitHub仓库获取的数据集
│   ├── dataset/
│   │   └── images/
│   │       └── vat_train/              # 包含发票图像的目录
│   └── label-data-for-nova-custom-fine-tunning/
│       └── output/                     # 训练JSON文件的输出目录
│
├── output/                             # 输出目录
│   ├── logs/                           # 日志文件目录
│   │   ├── nova_data_preparation.log   # 数据准备过程的日志
│   │   ├── upload_training_data.log    # 上传训练数据的日志
│   │   ├── nova_validation.log         # 数据验证的日志
│   │   └── complete_pipeline.log       # 完整流水线的日志
│   └── models/                         # 模型输出目录
│
├── docs/                               # 文档目录
│   ├── training_loss_plot.png          # 训练损失图表
│   ├── detailed_training_metrics.png   # 详细训练指标图表
│   ├── api_reference.md                # API参考文档
│   └── pipeline_usage.md               # 流水线使用指南
│
└── config.env                          # 环境变量配置文件
```

The `InvoiceDatasets` directory is sourced from a GitHub repository: https://github.com/FuxiJia/InvoiceDatasets.git

## Prerequisites

1. AWS CLI installed and configured with appropriate permissions
2. Python 3.6+ with the following packages:
   - boto3
   - pandas
   - python-dotenv
   - matplotlib
   - seaborn
   - pydantic

## Setup

1. Make sure your AWS credentials are configured:
   ```
   aws configure
   ```

2. Make the shell scripts executable:
   ```
   chmod +x scripts/*.sh
   ```

3. Set up the environment:
   ```
   cd scripts
   ./setup_environment.sh
   ```

## Configuration

The project uses a central configuration file `config.env` that contains all path and AWS settings. You can modify this file to customize:

- Local directory paths
- S3 bucket and prefix paths
- AWS account ID
- Log file locations

## Usage

### 方法1: 使用完整流水线（推荐）

运行完整的数据处理流水线：
```
cd scripts
./run_complete_pipeline.sh [S3_BUCKET]
```

如果需要包含LLM标注生成步骤：
```
cd scripts
./run_complete_pipeline.sh [S3_BUCKET] --generate-labels
```

完整流水线将执行以下步骤：
1. (可选) 使用LLM生成CSV标注数据
2. 处理图像并创建训练数据
3. 验证训练数据格式
4. 上传JSONL到S3

详细使用说明请参考 [流水线使用指南](./docs/pipeline_usage.md)。

### 方法2: 单独执行各步骤

#### Step 1: 生成标注数据（可选）

使用LLM生成标注数据：
```
cd scripts
python3 generate_labels_with_llm.py
```

#### Step 2: 准备训练数据

运行准备脚本：
```
cd scripts
./run_data_preparation.sh
```

该脚本将：
- 读取CSV文件中的图像名称和销售方信息
- 将图像上传到S3存储桶的 `nova-ft/images/` 前缀下
- 在输出目录中创建训练JSON文件
- 将处理过程记录在 `output/logs/nova_data_preparation.log`

#### Step 3: 上传训练JSON文件到S3

使用上传脚本将训练JSON文件发送到S3：
```
cd scripts
python3 upload_data_to_s3.py [--s3-bucket BUCKET_NAME]
```

#### Step 4: 创建微调作业

使用AWS CLI创建微调作业：
```
aws bedrock create-model-customization-job \
  --customization-type FINE_TUNING \
  --base-model-identifier anthropic.claude-3-sonnet-20240229-v1:0 \
  --job-name "invoice-seller-extraction" \
  --role-arn "arn:aws:iam::390468416359:role/service-role/AmazonBedrockExecutionRoleForNova" \
  --custom-model-name "invoice-seller-extraction" \
  --training-data-config "s3Uri=s3://your-bucket/nova-ft/training/data/" \
  --hyperparameters "epochCount=3,batchSize=1,learningRate=0.0001" \
  --output-data-config "s3Uri=s3://your-bucket/nova-ft/output/"
```

## Training Data Format

The training data follows the Amazon Bedrock Nova fine-tuning format:

```json
{
  "schemaVersion": "bedrock-conversation-2024",
  "system": [{
    "text": "You are a smart assistant that answers questions respectfully"
  }],
  "messages": [{
      "role": "user",
      "content": [{
          "text": "这是一张发票图片。请识别并提取出销售方名称。只需要返回销售方名称，不要有其他文字。请确保提取的是销售方（开票方），而不是购买方（收票方）。"
        },
        {
          "image": {
            "format": "jpg",
            "source": {
              "s3Location": {
                "uri": "your-bucket/nova-ft/images/vat_xxxx.jpg",
                "bucketOwner": "390468416359"
              }
            }
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [{
        "text": "销售方名称"
      }]
    }
  ]
}
```

## References

- [Amazon Bedrock Nova Fine-tuning Documentation](https://docs.aws.amazon.com/nova/latest/userguide/fine-tune-prepare-data-understanding.html)
- [Training Data Validation Tool](https://github.com/aws-samples/amazon-bedrock-samples/blob/main/custom-models/bedrock-fine-tuning/nova/understanding/dataset_validation/nova_ft_dataset_validator.py)

## Training Results

![training loss](./docs/training_loss_plot.png)
