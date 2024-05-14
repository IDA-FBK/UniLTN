import argparse
from plots import plot_metric_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-example", type=str, default="multilabel", help="specify the example (multilabel, mnist_single_digit_addition)")
    parser.add_argument("-path_to_results", type=str, default="./results/multilabel/",
        help="directory where to store the results")
    parser.add_argument("-exp_id", type=str, default="seed-0_epochs500_lr0.001", help="identifier for exp")
    parser.add_argument("-metric", type=str, default="Satisfiability", help="metrics to plot, options = (Satisifiability, Loss, Accuracy)")

    args = parser.parse_args()
    example = args.example
    path_to_results = args.path_to_results
    exp_id = args.exp_id
    metric = args.metric

    plot_metric_csv(example, path_to_results, exp_id, metric)

