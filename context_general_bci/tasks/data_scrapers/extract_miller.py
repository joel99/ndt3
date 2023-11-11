import os
import subprocess
import argparse

def extract_7z_files(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.7z'):
            stem_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(directory_path, stem_name)

            # Create the output directory if it doesn't exist
            if os.path.exists(output_dir):
                print(f"Output directory {output_dir} already exists, skipping.")
                continue
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            input_file_path = os.path.join(directory_path, filename)

            # Run the 7za command to extract the file
            command = f'7za e {input_file_path} -o{output_dir}'
            subprocess.run(command, shell=True)

def main():
    parser = argparse.ArgumentParser(description="Extract .7z files in a directory to their corresponding stems.")
    parser.add_argument("directory", help="Path to the directory containing .7z files.")
    args = parser.parse_args()

    extract_7z_files(args.directory)

if __name__ == "__main__":
    main()
