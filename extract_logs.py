import os
import sys

def extract_first_n_lines(input_path: str, output_path: str, line_count: int = 1000):
    """
    Extracts the exact first `line_count` lines from `input_path` and writes them
    to `output_path` without any modifications. Uses binary mode to preserve 
    exact byte sequences and line endings (\n or \r\n).
    """
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        sys.exit(1)

    lines_written = 0
    with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
        for line in infile:
            outfile.write(line)
            lines_written += 1
            if lines_written >= line_count:
                break

    print(f"Successfully copied {lines_written} lines from '{input_path}' to '{output_path}'.")

if __name__ == "__main__":
    # Support command line arguments or default fallback
    input_file = sys.argv[1] if len(sys.argv) > 1 else "HDFS.log"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "HDFS_first_1000.log"
    num_lines = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    extract_first_n_lines(input_file, output_file, num_lines)
