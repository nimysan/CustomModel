#!/usr/bin/env python3
import os
import argparse
import subprocess
import logging
import glob
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nova_validation.log"),
        logging.StreamHandler()
    ]
)

# Default configuration
DEFAULT_CONFIG = {
    "input_dir": "/Users/yexw/PycharmProjects/nova-fine-tunning/InvoiceDatasets/label-data-for-nova-custom-fine-tunning/output",
    "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
    "validator_path": "./nova_ft_dataset_validator.py"
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate Nova fine-tuning dataset')
    
    parser.add_argument('--input-dir', type=str, default=DEFAULT_CONFIG["input_dir"],
                        help=f'Directory containing JSON training files (default: {DEFAULT_CONFIG["input_dir"]})')
    
    parser.add_argument('--model-name', type=str, default=DEFAULT_CONFIG["model_name"],
                        help=f'Model name for validation (default: {DEFAULT_CONFIG["model_name"]})')
    
    parser.add_argument('--validator-path', type=str, default=DEFAULT_CONFIG["validator_path"],
                        help=f'Path to nova_ft_dataset_validator.py (default: {DEFAULT_CONFIG["validator_path"]})')
    
    parser.add_argument('--single-file', type=str, default=None,
                        help='Validate a single file instead of a directory')
    
    return parser.parse_args()

def validate_single_file(file_path, model_name, validator_path):
    """Validate a single JSON file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return False
    
    try:
        logging.info(f"Validating file: {file_path}")
        cmd = ["python3", validator_path, "-i", file_path, "-m", model_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"Validation successful: {file_path}")
            if result.stdout:
                logging.info(f"Output: {result.stdout}")
            return True
        else:
            logging.error(f"Validation failed for {file_path}: {result.stderr}")
            if result.stdout:
                logging.info(f"Output: {result.stdout}")
            return False
    
    except Exception as e:
        logging.error(f"Error validating file {file_path}: {e}")
        return False

def validate_directory(input_dir, model_name, validator_path):
    """Validate all JSON files in a directory."""
    if not os.path.exists(input_dir):
        logging.error(f"Directory not found: {input_dir}")
        return
    
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not json_files:
        logging.error(f"No JSON files found in {input_dir}")
        return
    
    logging.info(f"Found {len(json_files)} JSON files to validate")
    
    # Validate each file
    successful = 0
    failed = 0
    
    for file_path in json_files:
        if validate_single_file(file_path, model_name, validator_path):
            successful += 1
        else:
            failed += 1
    
    # Log summary
    logging.info(f"Validation completed. Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.warning(f"⚠️ {failed} files failed validation. Check the log for details.")
    else:
        logging.info("✅ All files passed validation!")

def check_validator_exists(validator_path):
    """Check if the validator script exists."""
    if os.path.exists(validator_path):
        return True
    
    # If not found at the specified path, check if it's in the current directory
    current_dir_path = os.path.join(os.getcwd(), os.path.basename(validator_path))
    if os.path.exists(current_dir_path):
        return True
    
    logging.error(f"Validator script not found at {validator_path}")
    logging.error("Please download the validator script from: https://github.com/aws-samples/amazon-bedrock-samples/blob/main/custom-models/bedrock-fine-tuning/nova/understanding/dataset_validation/nova_ft_dataset_validator.py")
    return False

def main():
    """Main function to validate Nova fine-tuning dataset."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if validator exists
    if not check_validator_exists(args.validator_path):
        return
    
    # Log the configuration
    logging.info(f"Validator path: {args.validator_path}")
    logging.info(f"Model name: {args.model_name}")
    
    # Validate files
    if args.single_file:
        validate_single_file(args.single_file, args.model_name, args.validator_path)
    else:
        logging.info(f"Input directory: {args.input_dir}")
        validate_directory(args.input_dir, args.model_name, args.validator_path)

if __name__ == "__main__":
    main()
