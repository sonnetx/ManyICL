import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
import csv
import re

def parse_response(response):
    """
    Parse the response to extract the answer choice and confidence score.
    """
    # More flexible regex for answer choice
    answer_match = re.search(r"Answer Choice \d+:\s*(?:\[?'?)(\w+)(?:'?\]?)", response)
    confidence_match = re.search(r'Confidence Score \d+: (0\.\d+)', response)
    
    answer = answer_match.group(1) if answer_match else "N/A"
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
    
    return answer, confidence

def pickle_to_csv(pickle_filename, csv_filename):
    """
    Parse an existing pickle file and save its contents to a CSV file.
    """
    with open(pickle_filename, 'rb') as f:
        results = pickle.load(f)
    
    save_results_to_csv(results, csv_filename)

def save_results_to_csv(results, csv_filename):
    """
    Save results dictionary to a CSV file.
    """
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question ID', 'Response', 'Answer', 'Confidence Score'])
        for qns_id, response in results.items():
            if qns_id != 'token_usage':
                answer, confidence = parse_response(response)
                writer.writerow([qns_id[1:-1], response, answer, confidence])

def parse_existing_pickles_to_csv(directory):
    """
    Parse all existing pickle files in a directory and save them as CSV files.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            pickle_path = os.path.join(directory, filename)
            csv_path = os.path.join(directory, filename.replace(".pkl", ".csv"))
            pickle_to_csv(pickle_path, csv_path)
            print(f"Converted {pickle_path} to {csv_path}")

parse_existing_pickles_to_csv("/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL")