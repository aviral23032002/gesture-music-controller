import pandas as pd
import os
import glob
import argparse

def fix_file(filepath):
    print(f"Fixing {filepath}...")
    # Read the file, detecting separator automatically
    df = pd.read_csv(filepath, sep=None, engine="python")
    
    # Store original column names to preserve them
    original_cols = df.columns.tolist()
    
    # Mapping for case-insensitive column matching
    col_mapping = {c.upper(): c for c in original_cols}
    
    # Invert X and Y axes (multiply by -1)
    # This applies to both Accelerometer (AX, AY) and Gyroscope (GX, GY)
    for axis in ['AX', 'AY', 'GX', 'GY']:
        if axis in col_mapping:
            actual_col = col_mapping[axis]
            df[actual_col] = df[actual_col] * -1
            
    # Save back to the same file, overwriting it
    df.to_csv(filepath, index=False, sep=' ')
    print(f"  -> Saved.")

def main():
    parser = argparse.ArgumentParser(description="Fix inverted X and Y axes in gesture recordings.")
    parser.add_argument("path", help="Path to a specific .txt file or a directory containing .txt files")
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        fix_file(args.path)
    elif os.path.isdir(args.path):
        files = []
        for root, _, filenames in os.walk(args.path):
            for filename in filenames:
                if filename.endswith('.txt'):
                    files.append(os.path.join(root, filename))
                    
        if not files:
            print(f"No .txt files found in {args.path} or its subdirectories.")
            return
        
        print(f"Found {len(files)} files. Inverting X and Y axes...")
        for f in files:
            fix_file(f)
        print(f"Successfully fixed {len(files)} files.")
    else:
        print("Error: Invalid path. Please provide a valid file or directory.")

if __name__ == "__main__":
    main()
