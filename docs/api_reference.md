# API Reference for Nova Fine-tuning Scripts

This document provides detailed information about the scripts used in the Nova fine-tuning project.

## prepare_nova_training_data.py

Processes invoice images and creates training data in the format required by Amazon Bedrock Nova.

### Usage

```bash
python3 scripts/prepare_nova_training_data.py
```

### Configuration

- `CSV_PATH`: Path to the CSV file containing image names and seller information
- `IMAGES_DIR`: Directory containing invoice images
- `S3_BUCKET`: S3 bucket name for uploading images
- `S3_PREFIX`: S3 prefix (folder path) for uploaded images
- `ACCOUNT_ID`: AWS account ID
- `OUTPUT_DIR`: Directory for output JSON files
- `OUTPUT_JSONL`: Path to the output JSONL file

## upload_training_data.py

Uploads training data to S3 for use with Amazon Bedrock Nova fine-tuning.

### Usage

```bash
python3 scripts/upload_training_data.py [options]
```

### Options

- `--input-dir`: Directory containing training files
- `--s3-bucket`: S3 bucket name
- `--s3-prefix`: S3 prefix (folder path)
- `--region`: AWS region
- `--dry-run`: Print files to upload without actually uploading

## run_nova_preparation.sh

Shell script to run the data preparation process.

### Usage

```bash
./scripts/run_nova_preparation.sh
```

## plot_training_metrics.py

Generates plots of training metrics from the fine-tuning job.

### Usage

```bash
python3 scripts/plot_training_metrics.py [options]
```

## nova_ft_dataset_validator.py

Validates the format of training data for Nova fine-tuning.

### Usage

```bash
python3 scripts/nova_ft_dataset_validator.py [options]
```
