import numpy as np
import pandas as pd
import os
import sys

def main():
    input_path = sys.argv[1]
    training_label = sys.argv[2]
    output_path = sys.argv[3]

    incorrect_index = incorrect_finder(input_path,training_label,output_path)
    incorrect_index.to_csv(output_path, index=False)

def read_labels_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]  # Remove newline characters
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return []
    except Exception as e:
         print(f"An error occurred: {e}")
         return []


def incorrect_finder(path,training_label,output_path):
    data_input = pd.read_csv(path, header=None)
    input_labels = data_input[0]

    training_output_labels = read_labels_file(training_label)
    training_output_labels = pd.Series(training_output_labels)

    non_matching_indices = input_labels[input_labels.astype(str) != training_output_labels.astype(str)].index



    non_matching_data = pd.DataFrame({
    "Index": non_matching_indices,
    "Input Label": input_labels.loc[non_matching_indices].values,
    "Output Label": training_output_labels.loc[non_matching_indices].values
})

    return non_matching_data

if __name__ == '__main__':
    main()
