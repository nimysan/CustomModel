#!/usr/bin/env python3
import boto3
import argparse
import logging
import time
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nova_finetuning_job.log"),
        logging.StreamHandler()
    ]
)

# Default configuration
DEFAULT_CONFIG = {
    "base_model_id": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0:300k",
    "job_name": f"invoice-seller-extraction-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "custom_model_name": "invoice-seller-extraction",
    "training_data_s3_uri": "s3://aigcdemo.plaza.red/nova-fine-tunning/training-data/training_data.jsonl",
    "output_s3_uri": "s3://aigcdemo.plaza.red/nova-fine-tunning/output/",
    "role_arn": "arn:aws:iam::390468416359:role/AmazonBedrockExecutionRoleForNova",
    "region": "us-east-1",
    "epoch_count": 3,
    "batch_size": 1,
    "learning_rate": 0.0001
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create Amazon Bedrock Nova fine-tuning job')
    
    parser.add_argument('--base-model-id', type=str, default=DEFAULT_CONFIG["base_model_id"],
                        help=f'Base model ID (default: {DEFAULT_CONFIG["base_model_id"]})')
    
    parser.add_argument('--job-name', type=str, default=DEFAULT_CONFIG["job_name"],
                        help=f'Job name (default: {DEFAULT_CONFIG["job_name"]})')
    
    parser.add_argument('--custom-model-name', type=str, default=DEFAULT_CONFIG["custom_model_name"],
                        help=f'Custom model name (default: {DEFAULT_CONFIG["custom_model_name"]})')
    
    parser.add_argument('--training-data-s3-uri', type=str, default=DEFAULT_CONFIG["training_data_s3_uri"],
                        help=f'S3 URI for training data (default: {DEFAULT_CONFIG["training_data_s3_uri"]})')
    
    parser.add_argument('--output-s3-uri', type=str, default=DEFAULT_CONFIG["output_s3_uri"],
                        help=f'S3 URI for output data (default: {DEFAULT_CONFIG["output_s3_uri"]})')
    
    parser.add_argument('--role-arn', type=str, default=DEFAULT_CONFIG["role_arn"],
                        help=f'IAM role ARN (default: {DEFAULT_CONFIG["role_arn"]})')
    
    parser.add_argument('--region', type=str, default=DEFAULT_CONFIG["region"],
                        help=f'AWS region (default: {DEFAULT_CONFIG["region"]})')
    
    parser.add_argument('--epoch-count', type=int, default=DEFAULT_CONFIG["epoch_count"],
                        help=f'Number of epochs (default: {DEFAULT_CONFIG["epoch_count"]})')
    
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG["batch_size"],
                        help=f'Batch size (default: {DEFAULT_CONFIG["batch_size"]})')
    
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help=f'Learning rate (default: {DEFAULT_CONFIG["learning_rate"]})')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the job configuration without creating the job')
    
    parser.add_argument('--skip-s3-check', action='store_true',
                        help='Skip checking S3 for training data')
    
    return parser.parse_args()

def check_s3_training_data(s3_uri, region):
    """Check if the S3 URI contains training files (JSON or JSONL)."""
    try:
        # Parse the S3 URI
        if not s3_uri.startswith('s3://'):
            logging.error(f"Invalid S3 URI format: {s3_uri}")
            return False
        
        parts = s3_uri[5:].split('/', 1)
        if len(parts) < 2:
            bucket = parts[0]
            prefix = ""
        else:
            bucket = parts[0]
            prefix = parts[1]
        
        # Create S3 client
        s3_client = boto3.client('s3', region_name=region)
        
        # List objects in the bucket with the given prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=10
        )
        
        # Check if there are any objects
        if 'Contents' not in response or len(response['Contents']) == 0:
            logging.error(f"No files found in S3 location: {s3_uri}")
            return False
        
        # Check if there are any JSON or JSONL files
        training_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.json') or obj['Key'].endswith('.jsonl')]
        if not training_files:
            logging.error(f"No JSON or JSONL files found in S3 location: {s3_uri}")
            return False
        
        # Log the found training files
        logging.info(f"Found {len(training_files)} training files in {s3_uri}")
        for i, file in enumerate(training_files[:5]):  # Show first 5 files
            logging.info(f"  {i+1}. {file['Key']} ({file['Size']} bytes)")
        
        if len(training_files) > 5:
            logging.info(f"  ... and {len(training_files) - 5} more files")
        
        return True
    
    except Exception as e:
        logging.error(f"Error checking S3 training data: {e}")
        return False

def create_fine_tuning_job(args):
    """Create a fine-tuning job using boto3."""
    try:
        # Check if training data exists in S3
        if not args.skip_s3_check:
            logging.info(f"Checking for training data in {args.training_data_s3_uri}...")
            if not check_s3_training_data(args.training_data_s3_uri, args.region):
                logging.error("No training data found. Please upload JSON or JSONL training files to the S3 location first.")
                logging.error("You can run the prepare_nova_training_data.py script to generate and upload training data.")
                logging.error("Or use --skip-s3-check to bypass this check.")
                return None
        
        # Create a boto3 client for Bedrock
        bedrock_client = boto3.client('bedrock', region_name=args.region)
        
        # Prepare hyperparameters
        hyperparameters = {
            "epochCount": str(args.epoch_count),
            "batchSize": str(args.batch_size),
            "learningRate": str(args.learning_rate)
        }
        
        # Prepare job configuration
        print(f"--------------------> {args.base_model_id}")
        job_config = {
            "customizationType": "FINE_TUNING",
            "baseModelIdentifier": args.base_model_id,
            "jobName": args.job_name,
            "customModelName": args.custom_model_name,
            "roleArn": args.role_arn,
            "trainingDataConfig": {
                "s3Uri": args.training_data_s3_uri
            },
            "outputDataConfig": {
                "s3Uri": args.output_s3_uri
            },
            "hyperParameters": hyperparameters
        }
        
        # Log the job configuration
        logging.info(f"Job configuration: {json.dumps(job_config, indent=2)}")
        
        # If dry run, just print the configuration and exit
        if args.dry_run:
            logging.info("Dry run mode. Job not created.")
            return None
        
        # Create the fine-tuning job
        response = bedrock_client.create_model_customization_job(**job_config)
        
        # Log the response
        job_id = response.get('jobArn', '').split('/')[-1]
        logging.info(f"Fine-tuning job created successfully. Job ID: {job_id}")
        logging.info(f"Job ARN: {response.get('jobArn')}")
        
        return response
    
    except Exception as e:
        logging.error(f"Error creating fine-tuning job: {e}")
        return None

def check_job_status(job_arn, region):
    """Check the status of a fine-tuning job."""
    try:
        bedrock_client = boto3.client('bedrock', region_name=region)
        response = bedrock_client.get_model_customization_job(jobIdentifier=job_arn)
        status = response.get('status', 'UNKNOWN')
        logging.info(f"Job status: {status}")
        return status
    except Exception as e:
        logging.error(f"Error checking job status: {e}")
        return "ERROR"

def main():
    """Main function to create a fine-tuning job."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create the fine-tuning job
    response = create_fine_tuning_job(args)
    
    if response and not args.dry_run:
        job_arn = response.get('jobArn')
        
        # Check initial status
        status = check_job_status(job_arn, args.region)
        
        # Print instructions for monitoring the job
        logging.info("\nFine-tuning job submitted successfully!")
        logging.info(f"Job ARN: {job_arn}")
        logging.info(f"Initial status: {status}")
        logging.info("\nTo monitor the job status:")
        logging.info(f"  1. Using AWS CLI: aws bedrock get-model-customization-job --job-identifier {job_arn} --region {args.region}")
        logging.info(f"  2. Using AWS Console: https://{args.region}.console.aws.amazon.com/bedrock/home?region={args.region}#/modelcustomization")

if __name__ == "__main__":
    main()
