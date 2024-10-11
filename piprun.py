import os
import subprocess

def read_file_content_with_fallback(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='ISO-8859-1') as f:
                return f.read()
        except UnicodeDecodeError:
            print(f"Skipping file due to encoding issues: {filepath}")
            return ""

def get_all_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                yield os.path.join(root, file)

def main():
    directory = '.'
    for filepath in get_all_files(directory):
        print(f"Reading file: {filepath}")
        content = read_file_content_with_fallback(filepath)
        # Here, content can be processed if needed.

    # Run pipreqs using subprocess to generate the requirements.txt file
    try:
        result = subprocess.run(
            ["pipreqs", directory, "--force", "--encoding", "utf-8"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running pipreqs: {e.stderr}")

if __name__ == "__main__":
    main()

