# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import shutil
import numbers
import pandas as pd
import matplotlib
from matplotlib import rc
 
BENCHMARKS_INPUT_FOLDER = './benchmarks'

sns.set(font_scale=1, rc={'text.usetex' : True})
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})


rc('font',**{'family':'serif','serif':['Times']})
matplotlib.rcParams['text.usetex'] = True


INSERTION_PLOT = "Insertion_Times"
SEARCH_PLOT = "Search_Times"
PRECISION_PLOT = "precision"

INSERTION_TIMES_FILENAME = "times_insertion"
SEARCH_PRECISION_KNN_TIMES_FILENAME = "search_knn"
SEARCH_PRECISION_PERCENTAGE_TIMES_FILENAME = "search_percentage"
PLOT_FILENAMES = [INSERTION_TIMES_FILENAME, SEARCH_PRECISION_KNN_TIMES_FILENAME, SEARCH_PRECISION_PERCENTAGE_TIMES_FILENAME]

DB_SIZES = [82533, 275110, 687777, 1375554, 2063331, 2751108]
CFGs = (4, 8, 8, 16), (16,32,32,64), (32, 64, 64, 128), \
          (64, 64, 128, 256), (128, 128, 256, 256), (128, 128, 256, 512)

GRAPH_DIR = "./graphs"

BOXPLOT_HORIZONTAL = "BOXPLOT_HORIZONTAL"
BOXPLOT_VERTICAL = "BOXPLOT_VERTICAL"
LINEAR_PLOT = "LINEAR_PLOT"
VIOLIN_PLOT = "VIOLIN_PLOT"

FILE_TIMES_VALUES_POSITION = 0
FILE_PRECISION_VALUES_POSITION = 1
MAX_KNN = 10

def create_plot(plot_title, file_prefix, db_size, _plot_type):
    fig, ax = plt.subplots()
    plot_files = []
    i = 0
    
    for _cfg in CFGs:
        i = i + 1
        subindices = ','.join(str(val) for val in _cfg)
        _cfg_str = 'CFG$_{\{' + subindices +  '\}}$'
        if file_prefix != INSERTION_TIMES_FILENAME:
            filename = file_prefix + "_" + "_".join(str(_str) for _str in _cfg) + f'_{db_size}.txt'
        else: 
            filename = file_prefix + "_" + "_".join(str(_str) for _str in _cfg) + f'_{db_size}.txt'
        plot_files.append(filename)
        data = np.loadtxt(f"{BENCHMARKS_INPUT_FOLDER}/{filename}")
           
        if i == 1:
            df = pd.DataFrame(data)
            df.columns = [_cfg_str]
        else:
            df[_cfg_str] = data

    if _plot_type == 'l':
        sns.lineplot(data=df, palette="Paired")
        _plot_str = "lineplot"
        ax.set_xlabel('Point')
        ax.set_ylabel('Time (s)')
    elif _plot_type.startswith('b'):
        sns.boxplot(data=df, palette="Paired", orient=_plot_type[1])
        _plot_str = "boxplot" + _plot_type[1]
        if _plot_type[1] == 'h':
            ax.set_ylabel('Configurations')
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xlabel('Configurations')
            ax.set_ylabel('Time (s)')

    elif _plot_type == 'v':
        sns.violinplot(data=df, palette="Paired", orient="h")
        _plot_str = "violinplot"
        ax.set_ylabel('Configurations')
        ax.set_xlabel('Time (s)')
    

    plt.savefig(os.path.join(GRAPH_DIR, file_prefix + f"_{db_size}_{_plot_str}.pdf"), bbox_inches='tight')
    plt.clf()


def create_insertion_times_plot(db_size):
    i = 0
    plot_files = []
    for _cfg in CFGs:
        i = i + 1
        subindices = ','.join(str(val) for val in _cfg)
        _cfg_str = 'CFG$_{\{' + subindices +  '\}}$'
        filename = INSERTION_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in _cfg) + f'_{db_size}.txt'
        plot_files.append(filename)
        data = np.loadtxt(f"{BENCHMARKS_INPUT_FOLDER}/{filename}")
        if i == 1:
            df = pd.DataFrame(data)
            df.columns = [_cfg_str]
        else:
            df[_cfg_str] = data

    create_boxplot(df, "h", INSERTION_TIMES_FILENAME, db_size, "Time (s)", "Configurations", "Dataset with " + str(db_size) + " entries")

def create_search_precision_plot(db_size, times_or_precision):
    i = 0
    plot_files = []
    for _cfg in CFGs:
        i = i + 1
        subindices = ','.join(str(val) for val in _cfg)
        _cfg_str = 'CFG$_{\{' + subindices +  '\}}$'
        filename = SEARCH_PRECISION_KNN_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in _cfg) + f'_{db_size}_1.txt'

        plot_files.append(filename)
        data = parse_values(filename, times_or_precision)
        if i == 1:
            df = pd.DataFrame(data)
            df.columns = [_cfg_str]
        else:
            df[_cfg_str] = data

    
    if times_or_precision == FILE_PRECISION_VALUES_POSITION:
        return create_boxplot(df, "h", "precision_" + SEARCH_PRECISION_KNN_TIMES_FILENAME, db_size, "Precision (Score)", "Configurations", "Dataset with " + str(db_size) + " entries")
    return create_boxplot(df, "h", SEARCH_PRECISION_KNN_TIMES_FILENAME, db_size, "Time (s)", "Configurations","Dataset with " + str(db_size) + " entries")

def create_precision_knn_plot(db_size):
    i = 0
    plot_files = []
    columns = ["configs"]
    for k in range (0, 10):
        columns.append("\\textsc{" + str(k+1) + "}")
    datas =  [[] for _ in range(0, 11)]
    for j, _cfg in enumerate(CFGs):
        subindices = ','.join(str(val) for val in _cfg)
        _cfg_str = 'CFG$_{\{' + subindices +  '\}}$'
        [datas[0].append(_cfg_str) for _ in range(1000)]
        for knn in range(MAX_KNN, 0, -1):
            i = i + 1
            filename = SEARCH_PRECISION_KNN_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in CFGs[j]) + f'_{db_size}_10.txt'

            plot_files.append(filename)
            data = parse_knn_values(filename, knn)
            data += [np.nan] * (1000 - len(data))
            datas[knn] += data
        i = 0
    

    df = pd.DataFrame(np.transpose(datas), columns=columns)
    return create_group_boxplot(df, "v", "precision_search_10knn", db_size, "KNN", "Precision (Score)", "Dataset with " + str(db_size) + " entries")


def create_precision_percentage_plot(db_size):
    datas =  []
    indexes = []
    for j in range(0, len(CFGs)):
        indexes.append("\\textsc{CFG" + str(j+1) + "}")
        config_data = []
        filename = SEARCH_PRECISION_PERCENTAGE_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in CFGs[j]) + f'_{db_size}_60.txt'
        data = parse_percentage_values(filename)
        for percentage_array in data:
            if len(config_data) < len(percentage_array) + 1:
                [config_data.append(0) for _ in range(0, len(percentage_array) - len(config_data) + 1)]
            config_data[len(percentage_array)] += 1 

        datas.append(config_data)

    df = pd.DataFrame(datas, index=indexes)
    df.index.name = 'Configuration'
    df_long = df.reset_index().melt(id_vars='Configuration')
    return create_scatterplot(df_long, "v", "precision_search_percentage", db_size, "No. Nodes", "Frequency")


def parse_values(filename, values_position):
    data = []
    with open(f"{BENCHMARKS_INPUT_FOLDER}/{filename}") as file:
        for line in file:
            value = float(line.split(': ')[values_position].replace('[','').replace(']',''))
            data.append(value)
    return data

def parse_knn_values(filename, knn):
    data = []
    with open(f"{BENCHMARKS_INPUT_FOLDER}/{filename}") as file:
        for line in file:
            numbers = eval(line.split(': ')[1])[:knn]
            data = data + numbers
    return data

def parse_percentage_values(filename):
    data = []

    with open(f"{BENCHMARKS_INPUT_FOLDER}/{filename}") as file:
        for line in file:
            numbers = eval(line.split(': ')[1])
            data.append(numbers)
    return data

def create_boxplot(df, plot_type, file_prefix, db_size, x_label, y_label, title):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, palette="Paired", orient=plot_type)
    _plot_str = "boxplot" + plot_type
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    plt.title(title)
    plt.savefig(os.path.join(GRAPH_DIR, file_prefix + f"_{db_size}_{_plot_str}.pdf"), bbox_inches='tight')
    plt.clf()

def create_group_boxplot(df, plot_type, file_prefix, db_size, x_label, y_label, title):
    fig, ax = plt.subplots()
    df = pd.melt(df, "configs")
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    sns.boxplot(data=df, hue="configs", x="variable", y="value", orient=plot_type)
    _plot_str = "boxplot" + plot_type
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    plt.title(title)
    plt.savefig(os.path.join(GRAPH_DIR, file_prefix + f"_{db_size}_{_plot_str}.pdf"), bbox_inches='tight')
    plt.clf()

def create_scatterplot(df, plot_type, file_prefix, db_size, x_label, y_label):
    fig, ax = plt.subplots()
    
    sns.scatterplot(data=df, x='variable', y='value',
                     style='Configuration', hue='Configuration', palette='dark')
    _plot_str = "boxplot" + plot_type
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    plt.savefig(os.path.join(GRAPH_DIR, file_prefix + f"_{db_size}_{_plot_str}.pdf"), bbox_inches='tight')
    plt.clf()


def create_plots():
    for size in DB_SIZES:
        create_insertion_times_plot(size) # Insertion
        create_search_precision_plot(size, FILE_TIMES_VALUES_POSITION) # KNN Seach times
        create_search_precision_plot(size, FILE_PRECISION_VALUES_POSITION) # KNN Search precision
        create_precision_knn_plot(size) # KNN Precision 1 to 10
        create_precision_percentage_plot(size) # Percentage search precision

if __name__ == "__main__":
    create_plots()
