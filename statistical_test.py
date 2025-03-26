import argparse
import os
import re

import pandas as pd
from scipy.stats import ranksums


def name_from_path(path: str) -> str:
    path_without_separators = re.sub(r"[\\/]", "_", path)
    path_simplified = path_without_separators.replace("results_", "")

    return path_simplified


def main(x_dir: str, y_dir: str, out_dir: str, level_of_significance: int) -> None:
    x_name = name_from_path(os.path.normpath(x_dir))
    y_name = name_from_path(os.path.normpath(y_dir))

    name = f"{x_name}_vs_{y_name}"

    x_contents = os.listdir(x_dir)
    y_contents = os.listdir(y_dir)

    datasets_to_compare = [x for x in x_contents if x in set(y_contents)]
    datasets_to_compare.remove("averages.csv")

    pvalues_results = []
    significant_differences_results = []

    shared_columns = []

    for dataset in datasets_to_compare:
        x_path = f"{x_dir}/{dataset}"
        y_path = f"{y_dir}/{dataset}"

        x_df = pd.read_csv(x_path)
        y_df = pd.read_csv(y_path)

        pvalues = []

        shared_columns = [x for x in x_df if x in set(y_df)]
        shared_columns.remove("Repetition")

        for column in shared_columns:
            ranksum = ranksums(x_df[column], y_df[column], alternative="two-sided")
            pvalue = ranksum[1]
            pvalues.append(pvalue)

        pvalues_results.append(pvalues)

        significant_differences = [pvalue < level_of_significance for pvalue in pvalues]
        significant_differences_results.append(significant_differences)

    os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    datasets = pd.Series(datasets_to_compare)

    pvalues_df = pd.DataFrame(
        pvalues_results,
        columns=shared_columns,
    )
    pvalues_df.insert(0, "Dataset", datasets)

    pvalues_df.to_csv(f"{out_dir}/{name}/pvalues.csv", index=False)

    significant_differences_df = pd.DataFrame(
        significant_differences_results,
        columns=shared_columns,
    )
    significant_differences_df.insert(0, "Dataset", datasets)

    significant_differences_df.to_csv(
        f"{out_dir}/{name}/significant_differences.csv",
        index=False,
    )

    print(pvalues_df)
    print(significant_differences_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Statistical Test", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "x_dir",
        help="First results directory",
    )
    parser.add_argument(
        "y_dir",
        help="Second results directory",
    )
    parser.add_argument("--out-dir", default="statistical_tests", help="Directory to save the results")
    parser.add_argument("--level-of-significance", default=0.05, type=int, help="Level of significance")

    args = parser.parse_args()

    main(args.x_dir, args.y_dir, args.out_dir, args.level_of_significane)
