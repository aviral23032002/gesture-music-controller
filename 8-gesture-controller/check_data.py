import os

def main():
    data_dir = "/Users/danny/Downloads/data"
    print(f"Checking data directory: {data_dir}")
    if not os.path.exists(data_dir):
        print("Error: Directory does not exist.")
        return
    
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print("\nGesture counts:")
    print("-" * 40)
    for sub in subdirs:
        subpath = os.path.join(data_dir, sub)
        files = [f for f in os.listdir(subpath) if f.endswith(".txt")]
        print(f"  * {sub:20} : {len(files)} files")
    print("-" * 40)

if __name__ == "__main__":
    main()
