import pandas as pd, numpy as np
import argparse
import pickle
import ast


def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def convert_pkl(raw_pickle, answer_prefix="Answer Choice "):
    """
    Convert raw pickle files saved in run.py to answers to each individual question

    raw_pickle [dict]: Python dict read from raw pickle files
    answer_prefix [str]: Prefix for answer matching, it should match the template in prompt.py
    """
    results = {}

    def extract_ans(ans_str, search_substring):
        # Split the string into lines
        lines = ans_str.split("\n")

        for line in lines:
            # Check if the line starts with the specified substring
            if line.startswith(search_substring):
                # If it does, add it to the list of extracted rows
                return line[len(search_substring) :].strip()
        return "ERROR"  # Answer not found

    for k, v in raw_pickle.items():
        if k.startswith("["):  # Skip token_usage
            qns_idx = ast.literal_eval(k)
            for idx, qn_idx in enumerate(qns_idx):
                results[qn_idx] = extract_ans(
                    v, f"{answer_prefix}{idx+1}:"
                )  # We start with question 1
    return results


def cal_metrics(
    EXP_NAME, test_df, classes, class_desp, show_error=True, bootstrap=1000
):
    """
    Calculate accuracy from model responses

    EXP_NAME [str]: experiment name which should match the pkl file generated by run.py
    test_df [pandas dataframe]: dataframe for test cases
    classes [list of str]: names of categories for classification, and this should match tbe columns of test_df and demo_df.
    class_desp [list of str]: category descriptions for classification, and these are the actual options sent to the model
    show_error [bool]: whether parsing errors will be printed
    bootstrap [int]: number of replicates for bootstrapping standard deviation
    """

    with open(f"{EXP_NAME}.pkl", "rb") as f:
        results = pickle.load(f)
    results = convert_pkl(
        results
    )  # Convert the batched results into individual answers
    # df.to_csv('file1.csv')
    print(results)
    test_df = test_df[classes]
    test_df['malignant'] = test_df['malignant'].apply(pd.to_numeric, errors='coerce')
    test_df['benign'] = test_df['benign'].apply(pd.to_numeric, errors='coerce')
    test_df.set_index('DDI_file')
    # print(test_df) # this is the ground truth df

    category_labels = test_df.columns
    index_labels = category_labels.map(class_to_idx)
    # print('index_labels', index_labels) # index_labels Index([0, 1, 2], dtype='int64')

    num_errors = 0
    labels, preds = [], []
    for i in test_df.itertuples():
        if (i.Index not in results) or (results[i.Index].startswith("ERROR")):
            num_errors += 1
            if show_error:
                print(i.Index, f"answer not found")
            continue

        pred_text = results[i.Index]
        pred_idx = len(classes)
        for idx, class_name in enumerate(class_desp):
            if class_name.lower() in pred_text.lower():
                if pred_idx == len(classes):
                    pred_idx = idx
                else:
                    pred_idx = -1  # Multiple predictions found
        # print(i) # Pandas(Index=392, DDI_file='000393.png', malignant=0, benign=1)
        if pred_idx >= 0:
            print(pred_idx)
            field_with_one = [field_name for field_name, field_value in i._asdict().items() if field_value == 1][0]
            # labels.append(index_labels.loc[i.Index])
            print(field_with_one)
            if field_with_one == "benign":
                labels.append(2)
            else:
                labels.append(1)
            preds.append(pred_idx)
        else:
            if show_error:
                print(
                    i.Index, f"multiple predictions found. raw response = {pred_text}"
                )
            num_errors += 1

    print(f"{EXP_NAME} In total {num_errors} errors len = {len(labels)}")

    y_true = np.array(labels)
    y_pred = np.array(preds)
    ori_acc = 100 * accuracy_score(y_true, y_pred)
    accs = []
    for boot in range(bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        accs.append(100 * accuracy_score(y_true[idx], y_pred[idx]))
    print(f"Accuracy: {ori_acc:.2f} +- {np.std(accs):.2f}")


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Experiment script.")
    # Adding the arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="DDI",
        help="The dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="Gemini1.5",
        help="The model to use",
    )
    parser.add_argument(
        "--location",
        type=str,
        required=False,
        default="us-central1",
        help="The location for the experiment",
    )
    parser.add_argument(
        "--num_shot_per_class",
        type=int,
        required=False,
        default=0,
        help="The number of shots per class",
    )
    parser.add_argument(
        "--num_qns_per_round",
        type=int,
        required=False,
        default=1,
        help="The number of questions asked each time",
    )

    # Parsing the arguments
    args = parser.parse_args()

    # Using the arguments
    dataset_name = args.dataset
    model = args.model
    location = args.location
    num_shot_per_class = args.num_shot_per_class
    num_qns_per_round = args.num_qns_per_round

    # Read the two dataframes for the dataset
    demo_df = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/demo.csv", index_col=0)
    test_df = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/test.csv", index_col=0)

    classes = list(demo_df.columns)  # classes for classification
    class_desp = classes  # The actual list of options given to the model. If the column names are informative enough, we can just use them.
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    EXP_NAME = f"{dataset_name}_{num_shot_per_class*len(classes)}shot_{model}_{num_qns_per_round}"
    cal_metrics(EXP_NAME, test_df, classes, class_desp)
