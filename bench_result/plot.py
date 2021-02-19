import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re


def plot_bench_result(
    dataset_names, baseline_results, tvm_results, algo_name, tvm_label, title=""
):
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("Paired")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_position = np.arange(len(dataset_names))
    ax.bar(x_position, baseline_results / tvm_results, width=0.4, label=algo_name)
    ax.bar(x_position + 0.4, np.ones((len(tvm_results))), width=0.4, label=tvm_label)
    ax.legend()
    ax.set_xticks(x_position + 0.2)
    ax.set_xticklabels(dataset_names)
    plt.title(title)
    out_file = algo_name + "_" + title.replace(", ", "_") + ".png"
    fig.savefig(out_file)
    # plt.show()


def do_plot(result_file, baseline, dataset_names, title, tvm_label):
    with open(result_file, "r") as f:
        result = json.load(f)

    algo_name_labels = {"rf": "sklearn-rf", "lgbm": "lgbm", "xgb": "xgb-gpu"}

    baseline_results = []
    tvm_results = []

    for dataset in dataset_names:
        d = result[dataset][baseline]
        baseline_results.append(float(d["prediction_time"]))
        tvm_results.append(float(d["hb-tvm"]["prediction_time"]))

    plot_bench_result(
        dataset_names,
        np.array(baseline_results),
        np.array(tvm_results),
        algo_name_labels[baseline],
        tvm_label,
        title,
    )


def plot_cpu():
    dataset_names = ["fraud", "epsilon", "year", "covtype", "higgs"]
    for result_file in ["result-cpu-500-8-10000.json", "result-cpu-500-8-50000.json"]:
        ntrees, maxdepth, batch = [int(i) for i in re.findall(r"\d+", result_file)]
        title = "batch={},ntrees={},maxdepth={}".format(batch, ntrees, maxdepth)
        do_plot(result_file, "rf", dataset_names, title, "tvm_cpu")


def plot_gpu():
    dataset_names = ["fraud", "year", "covtype", "higgs"]
    for result_file in [
        "result-gpu-500-12-10000.json",
        "result-gpu-500-8-1000.json",
        "result-gpu-500-8-10000.json",
    ]:
        ntrees, maxdepth, batch = [int(i) for i in re.findall(r"\d+", result_file)]
        title = "batch={},ntrees={},maxdepth={}".format(batch, ntrees, maxdepth)
        do_plot(result_file, "xgb", dataset_names, title, "tvm_gpu")


plot_cpu()
plot_gpu()
