# API Reference for Nova Fine-tuning Scripts

This document provides detailed information about the scripts used in the Nova fine-tuning project.

## run_complete_pipeline.sh

Shell script that executes the complete data processing pipeline from LLM labeling to S3 upload.

### Usage

```bash
./scripts/run_complete_pipeline.sh [options]
```

### Options

- `--generate-labels`, `-g`: Include the LLM label generation step (optional)

### Process Flow

1. (Optional) Generate CSV labels using LLM
2. Process images and create training data
3. Validate training data format
4. Upload JSONL to S3

## process_images_for_training.py

Processes invoice images and creates training data in the format required by Amazon Bedrock Nova.

### Usage

```bash
python3 scripts/process_images_for_training.py
```

### Configuration

- `CSV_PATH`: Path to the CSV file containing image names and seller information
- `IMAGES_DIR`: Directory containing invoice images
- `S3_BUCKET`: S3 bucket name for uploading images
- `S3_PREFIX`: S3 prefix (folder path) for uploaded images
- `ACCOUNT_ID`: AWS account ID
- `OUTPUT_DIR`: Directory for output JSON files
- `OUTPUT_JSONL`: Path to the output JSONL file

## upload_data_to_s3.py

Uploads training data to S3 for use with Amazon Bedrock Nova fine-tuning.

### Usage

```bash
python3 scripts/upload_data_to_s3.py [options]
```

### Options

- `--input-dir`: Directory containing training files
- `--s3-bucket`: S3 bucket name
- `--s3-prefix`: S3 prefix (folder path)
- `--region`: AWS region
- `--dry-run`: Print files to upload without actually uploading

## generate_labels_with_llm.py

Uses a large language model (LLM) to generate labels for invoice images, creating annotated data for fine-tuning.

### Usage

```bash
python3 scripts/generate_labels_with_llm.py [options]
```

### Options

- `--input-dir`: Directory containing invoice images
- `--output-file`: Path to output CSV file with generated labels
- `--model`: LLM model to use for label generation
- `--batch-size`: Number of images to process in each batch

## visualize_training_metrics.py

Generates plots of training metrics from the fine-tuning job.

### Usage

```bash
python3 scripts/visualize_training_metrics.py [options]
```

### Options

- `--metrics-file`: Path to the metrics CSV file
- `--output-dir`: Directory to save the generated plots
- `--show`: Display plots instead of saving them

## visualize_detailed_metrics.py

Generates detailed plots of training metrics with additional analysis.

### Usage

```bash
python3 scripts/visualize_detailed_metrics.py [options]
```

### Options

- `--metrics-file`: Path to the metrics CSV file
- `--output-dir`: Directory to save the generated plots
- `--show`: Display plots instead of saving them
- `--include-outliers`: Include outlier data points in the analysis

## nova_ft_dataset_validator.py

Validates the format of training data for Nova fine-tuning.

### Usage

```bash
python3 scripts/nova_ft_dataset_validator.py [options]
```

### Options

- `--input-file`: Path to the training data file to validate
- `--schema-version`: Schema version to validate against
- `--verbose`: Print detailed validation information

## validate_training_dataset.py

Validates the content and structure of the training dataset.

### Usage

```bash
python3 scripts/validate_training_dataset.py [options]
```

### Options

- `--input-dir`: Directory containing the training dataset
- `--report-file`: Path to save the validation report
- `--fix`: Attempt to fix common issues in the dataset

## run_data_preparation.sh

Shell script to run the data preparation process.

### Usage

```bash
./scripts/run_data_preparation.sh
```
