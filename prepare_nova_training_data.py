#!/usr/bin/env python3
import os
import csv
import json
import boto3
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nova_data_preparation.log"),
        logging.StreamHandler()
    ]
)

# Configuration
CSV_PATH = "/Users/yexw/PycharmProjects/nova-fine-tunning/invoice_sellers.csv"
IMAGES_DIR = "/Users/yexw/PycharmProjects/nova-fine-tunning/InvoiceDatasets/dataset/images/vat_train"
S3_BUCKET = "aigcdemo.plaza.red"
S3_PREFIX = "nova-fine-tunning/invoices/chinese"
ACCOUNT_ID = "390468416359"
OUTPUT_DIR = "/Users/yexw/PycharmProjects/nova-fine-tunning/InvoiceDatasets/label-data-for-nova-custom-fine-tunning/output"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "training_data.jsonl")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_csv_data():
    """Read the CSV file containing image names and seller information."""
    data = []
    filtered_count = 0
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Filter out entries with extraction failures
                seller_name = row.get('销售方', '')
                if '提取失败' in seller_name or 'ThrottlingException' in seller_name:
                    logging.warning(f"Skipping failed extraction entry: {row['图片名称']}")
                    filtered_count += 1
                    continue
                data.append(row)
        
        logging.info(f"Successfully read {len(data)} valid entries from CSV (filtered out {filtered_count} failed extractions)")
        return data
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return []

def upload_image_to_s3(image_path, image_name):
    """Upload an image to S3 and return the S3 URI."""
    s3_client = boto3.client('s3')
    s3_key = f"{S3_PREFIX}/{image_name}"
    
    try:
        s3_client.upload_file(image_path, S3_BUCKET, s3_key)
        s3_uri = f"{S3_BUCKET}/{s3_key}"
        logging.info(f"Successfully uploaded {image_name} to S3: {s3_uri}")
        return s3_uri
    except Exception as e:
        logging.error(f"Error uploading {image_name} to S3: {e}")
        return None

def validate_training_data(training_data):
    """Validate a training data object to ensure it meets Nova requirements."""
    try:
        # Check schema version
        if training_data.get('schemaVersion') != "bedrock-conversation-2024":
            logging.warning("Invalid schema version in training data")
            return False
        
        # Check system message
        if not training_data.get('system') or not isinstance(training_data.get('system'), list) or len(training_data.get('system')) == 0:
            logging.warning("Missing or invalid system message in training data")
            return False
        
        # Check messages
        if not training_data.get('messages') or not isinstance(training_data.get('messages'), list) or len(training_data.get('messages')) < 2:
            logging.warning("Missing or invalid messages in training data")
            return False
        
        # Check user message
        user_message = training_data.get('messages')[0]
        if user_message.get('role') != 'user' or not user_message.get('content'):
            logging.warning("Invalid user message in training data")
            return False
        
        # Check image in user message
        user_content = user_message.get('content', [])
        has_image = False
        for content_item in user_content:
            if 'image' in content_item:
                has_image = True
                image_data = content_item.get('image', {})
                if not image_data.get('format') or not image_data.get('source', {}).get('s3Location', {}).get('uri'):
                    logging.warning("Invalid image data in training data")
                    return False
        
        if not has_image:
            logging.warning("No image found in user message in training data")
            return False
        
        # Check assistant message
        assistant_message = training_data.get('messages')[1]
        if assistant_message.get('role') != 'assistant' or not assistant_message.get('content'):
            logging.warning("Invalid assistant message in training data")
            return False
        
        # Check assistant response
        assistant_content = assistant_message.get('content', [])
        if len(assistant_content) == 0 or not assistant_content[0].get('text'):
            logging.warning("Invalid assistant response in training data")
            return False
        
        return True
    
    except Exception as e:
        logging.error(f"Error validating training data: {e}")
        return False

def create_training_data(image_name, seller_name, s3_uri):
    """Create a training data object for Nova fine-tuning."""
    # Get the file extension (jpeg instead of jpg)
    file_format = image_name.split('.')[-1]
    
    training_data = {
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
                            "format": file_format,
                            "source": {
                                "s3Location": {
                                    "uri": s3_uri,
                                    "bucketOwner": ACCOUNT_ID
                                }
                            }
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{
                    "text": seller_name
                }]
            }
        ]
    }
    
    # Validate the created training data
    if validate_training_data(training_data):
        logging.info(f"Created and validated training data for: {image_name}")
        return training_data
    else:
        logging.error(f"Created but failed validation for training data: {image_name}")
        return None

def main():
    """Main function to process all images and create training data."""
    logging.info("Starting Nova fine-tuning data preparation")
    
    # Read CSV data
    csv_data = read_csv_data()
    if not csv_data:
        logging.error("No valid data found in CSV file. Exiting.")
        return
    
    # Process each entry
    successful_entries = 0
    failed_entries = 0
    skipped_entries = 0
    
    # Open the JSONL file for writing
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as jsonl_file:
        for entry in csv_data:
            try:
                image_name = entry['图片名称']
                seller_name = entry['销售方']
                
                # Additional validation for seller name
                if not seller_name or len(seller_name.strip()) == 0:
                    logging.warning(f"Empty seller name for {image_name}. Skipping.")
                    skipped_entries += 1
                    continue
                    
                # Check if image exists
                image_path = os.path.join(IMAGES_DIR, image_name)
                if not os.path.exists(image_path):
                    logging.warning(f"Image not found: {image_path}")
                    failed_entries += 1
                    continue
                
                # Upload image to S3
                s3_uri = upload_image_to_s3(image_path, image_name)
                if not s3_uri:
                    failed_entries += 1
                    continue
                
                # Create training data
                training_data = create_training_data(image_name, seller_name, s3_uri)
                if training_data:
                    # Write the training data as a single line in the JSONL file
                    jsonl_file.write(json.dumps(training_data, ensure_ascii=False) + '\n')
                    successful_entries += 1
                else:
                    failed_entries += 1
                    
            except Exception as e:
                logging.error(f"Error processing entry {entry}: {e}")
                failed_entries += 1
    
    logging.info(f"Data preparation completed.")
    logging.info(f"Successful: {successful_entries}, Failed: {failed_entries}, Skipped: {skipped_entries}")
    logging.info(f"Total processed: {successful_entries + failed_entries + skipped_entries}")
    
    # Print output file location
    if successful_entries > 0:
        logging.info(f"Training JSONL file is available at: {OUTPUT_JSONL}")
        logging.info("Next step: Upload this file to S3 using upload_training_data.py")

if __name__ == "__main__":
    main()
