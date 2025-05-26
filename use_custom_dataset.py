#!/usr/bin/env python3
"""
Example script showing how to use a custom CSV dataset with TabSTAR.

Usage:
    python use_custom_dataset.py --csv_path /path/to/your/dataset.csv --target_column target_col_name --pretrain_exp YOUR_PRETRAINED_MODEL

Requirements:
    1. Your CSV file should have a header row with column names
    2. One column should be designated as the target variable
    3. You need a pretrained TabSTAR model
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Use custom CSV dataset with TabSTAR")
    parser.add_argument('--csv_path', type=str, required=True, 
                       help='Path to your CSV file')
    parser.add_argument('--target_column', type=str, required=True,
                       help='Name of the target column in your CSV')
    parser.add_argument('--pretrain_exp', type=str, required=True,
                       help='Name of your pretrained TabSTAR model')
    parser.add_argument('--exp_name', type=str, default='custom_dataset_finetune',
                       help='Name for this finetuning experiment')
    parser.add_argument('--run_num', type=int, default=0,
                       help='Run number (for multiple runs)')
    parser.add_argument('--downstream_examples', type=int, default=1000,
                       help='Number of training examples to use')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    if not args.csv_path.endswith('.csv'):
        print(f"Error: File must be a CSV file: {args.csv_path}")
        sys.exit(1)
    
    # Quick validation of CSV structure
    try:
        import pandas as pd
        df = pd.read_csv(args.csv_path)
        if args.target_column not in df.columns:
            print(f"Error: Target column '{args.target_column}' not found in CSV.")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        print(f"✅ CSV validation passed:")
        print(f"   - File: {args.csv_path}")
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - Target: {args.target_column}")
        print(f"   - Features: {len(df.columns) - 1}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Construct the command to run do_finetune.py
    cmd = [
        'python', 'do_finetune.py',
        '--pretrain_exp', args.pretrain_exp,
        '--dataset_id', 'custom',
        '--custom_csv_path', args.csv_path,
        '--custom_target_column', args.target_column,
        '--exp', args.exp_name,
        '--run_num', str(args.run_num),
        '--downstream_examples', str(args.downstream_examples)
    ]
    
    print(f"\n🚀 Running TabSTAR finetuning with custom dataset...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Execute the command
    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Finetuning completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Finetuning failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 