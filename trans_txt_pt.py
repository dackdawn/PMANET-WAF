import random

def reduce_file_lines(input_file, output_file, fraction=0.2):
    """
    Reduces the number of lines in the input file by randomly selecting a fraction of them and writes to output file.

    Parameters:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        fraction (float): Fraction of lines to select (default is 0.2 for 1/5).
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    num_lines_to_keep = max(1, int(len(lines) * fraction))
    selected_lines = random.sample(lines, num_lines_to_keep)
    
    with open(output_file, 'w') as file:
        file.writelines(selected_lines)

if __name__ == '__main__':
    reduce_file_lines('benign_urls.txt', 'benign_urls_sample.txt')
    reduce_file_lines('malware_urls.txt', 'malware_urls_sample.txt')
