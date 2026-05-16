import os
import json
import glob

def merge_json_files(output_filename="electra_combined_results.json"):
    """
    Reads all JSON files in the current directory and merges them into a single JSON file.
    """
    # 1. Get the current directory (where the script is running)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Find all JSON files in this directory
    # We use glob to easily grab all files ending in .json
    json_files = glob.glob(os.path.join(current_dir, "*.json"))
    
    # 3. Create a dictionary to hold the combined results
    combined_data = {}
    
    print(f"🔍 Found {len(json_files)} JSON files in the current directory.")
    
    # 4. Loop through each file and extract the data
    for file_path in json_files:
        filename = os.path.basename(file_path)
        
        # Skip the output file if it already exists from a previous run
        if filename == output_filename:
            continue
            
        print(f"   -> Processing: {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # We use the filename (without extension) as the main key 
                # to keep the results organized and separated.
                task_name = os.path.splitext(filename)[0]
                combined_data[task_name] = data
                
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    # 5. Save the combined dictionary to a new JSON file
    output_path = os.path.join(current_dir, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # indent=4 makes the output file human-readable and cleanly formatted
            json.dump(combined_data, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Successfully merged all results into: {output_filename}")
        
    except Exception as e:
        print(f"\n❌ Error saving the combined file: {e}")

if __name__ == "__main__":
    merge_json_files()