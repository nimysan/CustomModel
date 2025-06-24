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
│   ├── jsonl_to_s3.py                  # 上传JSONL文件到S3的脚本
│   ├── generate_labels_with_llm.py     # 使用LLM生成标注数据的脚本
│   ├── visualize_training_metrics.py   # 生成训练指标图表的脚本
│   ├── visualize_detailed_metrics.py   # 生成详细训练指标图表的脚本
│   ├── nova_ft_dataset_validator.py    # 验证训练数据格式的脚本
│   ├── validate_jsonl.sh               # 验证JSONL文件的Shell脚本
│   ├── validate_training_dataset.py    # 验证训练数据集的脚本
│   ├── create_nova_ft_job.py           # 创建Nova微调作业的脚本
│   ├── run_data_preparation.sh         # 运行数据准备过程的Shell脚本
│   ├── run_complete_pipeline.sh        # 执行完整数据处理流水线的Shell脚本
│   └── setup_environment.sh            # 设置环境和目录结构的脚本
│
├── data/                               # 数据目录
│   ├── images/                         # 发票图像目录
│   │   ├── train/                      # 训练集图像目录
│   │   └── test/                       # 测试集图像目录
│   ├── label_data/                     # 标注数据目录
│   │   ├── train_label.csv             # 训练集标注数据
│   │   └── test_label.csv              # 测试集标注数据
│   └── bedrock-ft/                     # Bedrock微调数据目录
│       ├── train_data.jsonl            # 训练集JSONL文件
│       └── test_data.jsonl             # 测试集JSONL文件
│
├── output/                             # 输出目录
│   ├── logs/                           # 日志文件目录
│   │   ├── nova_data_preparation.log   # 数据准备过程的日志
│   │   ├── jsonl_to_s3.log             # 上传JSONL数据的日志
│   │   ├── nova_validation.log         # 数据验证的日志
│   │   ├── complete_pipeline.log       # 完整流水线的日志
│   │   ├── generate_labels.log         # 生成标注的日志
│   │   └── nova_finetuning_job.log     # 微调作业的日志
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
- AWS account ID and region
- Model parameters (epoch count, batch size, learning rate)
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

如果需要自动创建微调作业：
```
cd scripts
./run_complete_pipeline.sh [S3_BUCKET] --create-job
```

可以组合使用参数：
```
cd scripts
./run_complete_pipeline.sh [S3_BUCKET] --generate-labels --create-job
```

完整流水线将执行以下步骤：
1. (可选) 使用LLM生成训练集和测试集的CSV标注数据
2. 处理图像并创建训练集和测试集的JSONL文件
3. 验证JSONL文件格式
4. 上传JSONL文件到S3
5. (可选) 创建微调作业

详细使用说明请参考 [流水线使用指南](./docs/pipeline_usage.md)。

### 方法2: 单独执行各步骤

#### Step 1: 生成标注数据（可选）

使用LLM生成训练集和测试集的标注数据：
```
cd scripts
python3 generate_labels_with_llm.py
```

#### Step 2: 准备训练数据

处理图像并创建训练集和测试集的JSONL文件：
```
cd scripts
python3 process_images_for_training.py
```

#### Step 3: 验证JSONL文件

验证训练集和测试集的JSONL文件格式：
```
cd scripts
./validate_jsonl.sh
```

#### Step 4: 上传JSONL文件到S3

将JSONL文件上传到S3：
```
cd scripts
python3 jsonl_to_s3.py [--s3-bucket BUCKET_NAME]
```

#### Step 5: 创建微调作业

创建Nova微调作业：
```
cd scripts
python3 create_nova_ft_job.py [--s3-bucket BUCKET_NAME]
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
            "format": "jpeg",
            "source": {
              "s3Location": {
                "uri": "s3://your-bucket/nova-ft/images/vat_xxxx.jpeg",
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

- [Amazon Bedrock Nova Fine-tuning Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models-fine-tune.html)
- [Training Data Validation Tool](https://github.com/aws-samples/amazon-bedrock-samples/blob/main/custom-models/bedrock-fine-tuning/nova/understanding/dataset_validation/nova_ft_dataset_validator.py)

## Training Results

![training loss](./docs/training_loss_plot.png)
