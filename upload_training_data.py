#!/usr/bin/env python3
import os
import boto3
import argparse
import logging
import glob
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("upload_training_data.log"),
        logging.StreamHandler()
    ]
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Upload training data to S3 for Nova fine-tuning')
    
    parser.add_argument('--input-dir', type=str, 
                        default="/Users/yexw/PycharmProjects/nova-fine-tunning/InvoiceDatasets/label-data-for-nova-custom-fine-tunning/output",
                        help='Directory containing training files')
    
    parser.add_argument('--s3-bucket', type=str, 
                        default="aigcdemo.plaza.red",
                        help='S3 bucket name')
    
    parser.add_argument('--s3-prefix', type=str, 
                        default="nova-fine-tunning/training-data",
                        help='S3 prefix (folder path)')
    
    parser.add_argument('--region', type=str, 
                        default="us-east-1",
                        help='AWS region')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Print files to upload without actually uploading')
    
    return parser.parse_args()

def upload_files_to_s3(input_dir, s3_bucket, s3_prefix, region, dry_run=False):
    """Upload training files to S3."""
    try:
        # Check if input directory exists
        if not os.path.exists(input_dir):
            logging.error(f"Input directory does not exist: {input_dir}")
            return False
        
        # Find the JSONL file in the input directory
        jsonl_file = os.path.join(input_dir, "training_data.jsonl")
        if not os.path.exists(jsonl_file):
            logging.error(f"JSONL file not found: {jsonl_file}")
            return False
        
        logging.info(f"Found JSONL file to upload: {jsonl_file}")
        
        # Create S3 client
        s3_client = boto3.client('s3', region_name=region)
        
        # Upload the JSONL file
        file_name = os.path.basename(jsonl_file)
        s3_key = f"{s3_prefix}/{file_name}"
        
        if dry_run:
            logging.info(f"[DRY RUN] Would upload {jsonl_file} to s3://{s3_bucket}/{s3_key}")
            uploaded = True
        else:
            try:
                s3_client.upload_file(jsonl_file, s3_bucket, s3_key)
                logging.info(f"Uploaded {file_name} to s3://{s3_bucket}/{s3_key}")
                uploaded = True
            except Exception as e:
                logging.error(f"Error uploading {file_name}: {e}")
                uploaded = False
        
        # Verify upload
        if not dry_run and uploaded:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=s3_bucket,
                    Prefix=s3_prefix
                )
                
                if 'Contents' in response:
                    s3_files = [obj['Key'] for obj in response['Contents']]
                    logging.info(f"Verified files in S3 location: s3://{s3_bucket}/{s3_prefix}")
                    
                    # Print S3 URI for use with fine-tuning job
                    logging.info(f"\nTraining data is ready at: s3://{s3_bucket}/{s3_prefix}")
                    logging.info("You can now create a fine-tuning job using:")
                    logging.info(f"python3 create_nova_finetuning_job.py --training-data-s3-uri s3://{s3_bucket}/{s3_prefix}")
            except Exception as e:
                logging.error(f"Error verifying upload: {e}")
        
        return uploaded
    
    except Exception as e:
        logging.error(f"Error uploading file to S3: {e}")
        return False

def main():
    """Main function to upload training data."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Log the configuration
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"S3 bucket: {args.s3_bucket}")
    logging.info(f"S3 prefix: {args.s3_prefix}")
    logging.info(f"Region: {args.region}")
    logging.info(f"Dry run: {args.dry_run}")
    
    # Upload file to S3
    success = upload_files_to_s3(
        args.input_dir,
        args.s3_bucket,
        args.s3_prefix,
        args.region,
        args.dry_run
    )
    
    if success:
        logging.info("Upload completed successfully")
    else:
        logging.error("Upload failed")

if __name__ == "__main__":
    main()
