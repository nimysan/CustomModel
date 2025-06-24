#!/bin/bash

# Make sure AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
  echo "AWS credentials not configured. Please run 'aws configure' first."
  exit 1
fi

# Install required Python packages if not already installed
pip install boto3 pandas

# Run the data preparation script
echo "Starting data preparation for fine-tuning..."
python3 process_images_for_training.py

# Check if the script ran successfully
if [ $? -eq 0 ]; then
  echo "Data preparation completed successfully."
  echo "Training JSONL file is in: ../InvoiceDatasets/label-data-for-nova-custom-fine-tunning/output/training_data.jsonl"
  
  # Check if the JSONL file exists and count the number of lines
  JSONL_FILE="../InvoiceDatasets/label-data-for-nova-custom-fine-tunning/output/training_data.jsonl"
  if [ -f "$JSONL_FILE" ]; then
    LINE_COUNT=$(wc -l < "$JSONL_FILE")
    echo "Created JSONL file with $LINE_COUNT training examples."
  else
    echo "Warning: JSONL file not found."
  fi
  
  echo ""
  echo "Next steps for fine-tuning:"
  echo "1. Upload the JSONL file to an S3 bucket using the upload_data_to_s3.py script:"
  echo "   python3 upload_data_to_s3.py"
  echo ""
  echo "2. Create a fine-tuning job using the AWS CLI:"
  echo "aws bedrock create-model-customization-job \\"
  echo "  --customization-type FINE_TUNING \\"
  echo "  --base-model-identifier anthropic.claude-3-sonnet-20240229-v1:0 \\"
  echo "  --job-name \"invoice-seller-extraction\" \\"
  echo "  --role-arn \"arn:aws:iam::390468416359:role/service-role/AmazonBedrockExecutionRoleForNova\" \\"
  echo "  --custom-model-name \"invoice-seller-extraction\" \\"
  echo "  --training-data-config \"s3Uri=s3://aigcdemo.plaza.red/nova-fine-tunning/training-data/\" \\"
  echo "  --hyperparameters \"epochCount=3,batchSize=1,learningRate=0.0001\" \\"
  echo "  --output-data-config \"s3Uri=s3://aigcdemo.plaza.red/nova-fine-tunning/output/\""
else
  echo "Data preparation failed. Check the logs for details."
fi
