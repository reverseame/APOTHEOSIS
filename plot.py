# Import libraries
import matplotlib.pyplot as plt

plt.rcParams.update({#"figure.figsize" : [12, 10],
                     "text.usetex": True,
                     "font.family": "serif",
                     "font.serif": "Times",
                     "savefig.dpi": 130})

import numpy as np
import os
import seaborn as sns
import shutil
import numbers
import pandas as pd
import matplotlib
import statistics


from matplotlib import rc
from brokenaxes import brokenaxes

ALGORITHM = ["ssdeep", "tlsh"]

BENCHMARKS_INPUT_FOLDER = './benchmarks/'

#sns.set(font_scale=1, rc={'text.usetex' : True})
#sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
#matplotlib.rc('font',**{'family':'serif','serif':['Times']})
#matplotlib.rcParams['text.usetex'] = True

INSERTION_PLOT = "Insertion_Times"
SEARCH_PLOT = "Search_Times"
PRECISION_PLOT = "precision"

INSERTION_TIMES_FILENAME = "times_insertion"
SEARCH_PRECISION_KNN_TIMES_FILENAME = "search_knn"
SEARCH_PRECISION_PERCENTAGE_TIMES_FILENAME = "search_percentage"
BRUTEFORCE_PERCENTAGE_FILENAME = "hnsw"
BRUTEFORCE_FILENAME = "bruteforce"
PLOT_FILENAMES = [INSERTION_TIMES_FILENAME, SEARCH_PRECISION_KNN_TIMES_FILENAME, SEARCH_PRECISION_PERCENTAGE_TIMES_FILENAME]

CFGs = (4, 8, 8, 16), (16,32,32,64), (32, 64, 64, 128), \
          (64, 64, 128, 256), (128, 128, 256, 256), (128, 128, 256, 512)




M_PARAM_CFGs = (4, 8, 8, 8), (8, 8, 8, 8), (16, 8, 8, 8), (32, 8, 8, 8)
EF_PARAM_CFGs = (4, 8, 8, 8), (4, 16, 8, 8), (4, 32, 8, 8), (4, 64, 8, 8)
MMAX_PARAM_CFGs = (4, 8, 8, 8), (4, 8, 16, 8), (4, 8, 32, 8), (4, 8, 64, 8)
MMAX0_PARAM_CFGs = (4, 8, 8, 8), (4, 8, 8, 16), (4, 8, 8, 32), (4, 8, 8, 64)
INDIVIDUAL_PARAM_CFGs = [M_PARAM_CFGs, EF_PARAM_CFGs, MMAX_PARAM_CFGs, MMAX0_PARAM_CFGs]

PERCENTAGES = [3, 10, 25, 50, 75, 100]

GRAPH_DIR = "./graphs/" 

BOXPLOT_HORIZONTAL = "BOXPLOT_HORIZONTAL"
BOXPLOT_VERTICAL = "BOXPLOT_VERTICAL"
LINEAR_PLOT = "LINEAR_PLOT"
VIOLIN_PLOT = "VIOLIN_PLOT"

FILE_TIMES_VALUES_POSITION = 0
FILE_PRECISION_VALUES_POSITION = 1
MAX_KNN = 10

def create_insertion_times_plot(algo, db_size):
    i = 0
    plot_files = []
    for _cfg in CFGs:
        i = i + 1
        subindices = ','.join(str(val) for val in _cfg)
        _cfg_str = 'CFG$_{' + subindices +  '}$'
        filename = INSERTION_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in _cfg) + f'_{db_size}.txt'
        plot_files.append(filename)
        data = np.loadtxt(f"{BENCHMARKS_INPUT_FOLDER}/{algo}/{filename}")
        if i == 1:
            df = pd.DataFrame(data)
            df.columns = [_cfg_str]
        else:
            df[_cfg_str] = data
    create_boxplot(algo, df, "h", INSERTION_TIMES_FILENAME, db_size, "Time (s)", "Configurations", "Dataset with " + str(db_size) + " items")

def create_insertion_times_plot_summary(algorithm, db_sizes):
    lines = []
    for _cfg in CFGs:
        time_values = []
        for db_size in db_sizes:
            filename = INSERTION_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in _cfg) + f'_{db_size}.txt'
            data = np.loadtxt(f"{BENCHMARKS_INPUT_FOLDER}/{algorithm}/{filename}")
            time_values.append(sum(data))
        lines.append(time_values)

    for i, line in enumerate(lines):
        subscripts = ','.join(str(val) for val in CFGs[i])
        _cfg_str = 'CFG$_{' + subscripts +  '}$'
        plt.plot(PERCENTAGES, line, label =_cfg_str)

    #plt.title(f'Insertion times for {algorithm}')
    plt.xlabel(f'% of Dataset')
    plt.ylabel('Total time (s)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(os.path.join(GRAPH_DIR, algorithm), INSERTION_TIMES_FILENAME + f"_summary.pdf"), bbox_inches='tight')
    plt.clf()

def create_search_times_plot_summary(algorithm, db_sizes):
    lines = []
    for _cfg in CFGs:
        time_values = []
        for db_size in db_sizes:
            filename = SEARCH_PRECISION_KNN_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in _cfg) + f'_{db_size}_1.txt'
            data = parse_values(algorithm, filename, FILE_TIMES_VALUES_POSITION)
            time_values.append(sum(data))
        lines.append(time_values)

    for i, line in enumerate(lines):
        subscripts = ','.join(str(val) for val in CFGs[i])
        _cfg_str = 'CFG$_{' + subscripts +  '}$'
        plt.plot(PERCENTAGES, line, label =_cfg_str)

    plt.xlabel(f'% of Dataset')
    plt.ylabel('Total time (s)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(os.path.join(GRAPH_DIR, algorithm), SEARCH_PRECISION_KNN_TIMES_FILENAME + "_summary.pdf"), bbox_inches='tight')
    plt.clf()

def create_search_precision_plot_summary(algorithm, db_sizes):
    lines = []
    for _cfg in CFGs:
        time_values = []
        for db_size in db_sizes:
            filename = SEARCH_PRECISION_KNN_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in _cfg) + f'_{db_size}_1.txt'
            data = parse_values(algorithm, filename, FILE_PRECISION_VALUES_POSITION)
            time_values.append(statistics.mean(data))
        lines.append(time_values)

    for i, line in enumerate(lines):
        subscripts = ','.join(str(val) for val in CFGs[i])
        _cfg_str = 'CFG$_{' + subscripts +  '}$'
        plt.plot(PERCENTAGES, line, label =_cfg_str)

    plt.xlabel(f'% of Dataset')
    plt.ylabel('Avg. score')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(os.path.join(GRAPH_DIR, algorithm), "precision_" + SEARCH_PRECISION_KNN_TIMES_FILENAME + "_summary.pdf"), bbox_inches='tight')
    plt.clf()


def create_individual_parameter_plot_summary(algorithm, configs, parameter_name, x_data):
    line = []
    for _cfg in configs:
        filename = SEARCH_PRECISION_KNN_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in _cfg) + f'_82533_1.txt'
        data = parse_values(algorithm, filename, FILE_TIMES_VALUES_POSITION)
        line.append(sum(data))


    plt.plot(x_data, line)

    plt.xlabel(f'${parameter_name}$ Value')
    plt.ylabel('Time (s)')
    plt.grid(True)

    outfile = f"individual_times_{parameter_name}_summary.pdf"
    plt.savefig(os.path.join(os.path.join(GRAPH_DIR, algorithm), outfile), bbox_inches='tight')
    plt.clf()

def create_individual_parameter_precision_plot_summary(algorithm, configs, parameter_name, x_data):
    line = []
    for _cfg in configs:
        filename = SEARCH_PRECISION_KNN_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in _cfg) + f'_82533_1.txt'
        data = parse_values(algorithm, filename, FILE_PRECISION_VALUES_POSITION)
        line.append(statistics.mean(data))


    plt.plot(x_data, line)

    plt.xlabel(f'${parameter_name}$ Value')
    plt.ylabel('Avg. score')
    plt.grid(True)

    outfile = f"individual_precision_{parameter_name}_summary.pdf"
    plt.savefig(os.path.join(os.path.join(GRAPH_DIR, algorithm), outfile), bbox_inches='tight')
    plt.clf()


def create_search_precision_plot(algo, db_size, times_or_precision):
    i = 0
    plot_files = []
    for _cfg in CFGs:
        i = i + 1
        subindices = ','.join(str(val) for val in _cfg)
        _cfg_str = 'CFG$_{' + subindices +  '}$'
        filename = SEARCH_PRECISION_KNN_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in _cfg) + f'_{db_size}_1.txt'

        plot_files.append(filename)
        data = parse_values(algo, filename, times_or_precision)
        if i == 1:
            df = pd.DataFrame(data)
            df.columns = [_cfg_str]
        else:
            df[_cfg_str] = data

    
    if times_or_precision == FILE_PRECISION_VALUES_POSITION:
        return create_boxplot(algo, df, "h", "precision_" + SEARCH_PRECISION_KNN_TIMES_FILENAME, db_size, "Precision (score)", "Configurations", "Dataset with " + str(db_size) + " items")
    return create_boxplot(algo, df, "h", SEARCH_PRECISION_KNN_TIMES_FILENAME, db_size, "Time (s)", "Configurations","Dataset with " + str(db_size) + " items")

def create_precision_knn_plot(algo, db_size):
    i = 0
    plot_files = []
    columns = ["configs"]
    for k in range (0, 10):
        columns.append("\\textsc{" + str(k+1) + "}")
    datas =  [[] for _ in range(0, 11)]
    for j, _cfg in enumerate(CFGs):
        subindices = ','.join(str(val) for val in _cfg)
        _cfg_str = 'CFG$_{' + subindices +  '}$'
        [datas[0].append(_cfg_str) for _ in range(1000)]
        for knn in range(MAX_KNN, 0, -1):
            i = i + 1
            filename = SEARCH_PRECISION_KNN_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in CFGs[j]) + f'_{db_size}_10.txt'

            plot_files.append(filename)
            data = parse_knn_values(algo, filename, knn)
            data += [np.nan] * (1000 - len(data))
            datas[knn] += data
        i = 0

    df = pd.DataFrame(np.transpose(datas), columns=columns)
    return create_group_boxplot(algo, df, "v", "precision_search_10knn", db_size, "KNN", "Precision (score)", "Dataset with " + str(db_size) + " items")


def create_precision_percentage_plot(algo, db_size):
    datas =  []
    indexes = []
    for j in range(0, len(CFGs)):
        indexes.append("\\textsc{CFG" + str(j+1) + "}")
        config_data = []
        filename = SEARCH_PRECISION_PERCENTAGE_TIMES_FILENAME + "_" + "_".join(str(_str) for _str in CFGs[j]) + f'_{db_size}_60.txt'
        data = parse_percentage_values(algo, filename)
        for percentage_array in data:
            if len(config_data) < len(percentage_array) + 1:
                [config_data.append(0) for _ in range(0, len(percentage_array) - len(config_data) + 1)]
            config_data[len(percentage_array)] += 1 

        datas.append(config_data)

    df = pd.DataFrame(datas, index=indexes)
    df.index.name = 'Configuration'
    df_long = df.reset_index().melt(id_vars='Configuration')
    return create_scatterplot(algo, df_long, "v", "precision_search_percentage", db_size, "No. Nodes", "Frequency")

def create_bruteforce_vs_plot(algo, db_size):
    i = 0
    plot_files = []
    indexes = []
    times = []
    for i, _cfg in enumerate(CFGs, start=1):
        subindices = ','.join(str(val) for val in _cfg)
        _cfg_str = 'CFG$_{' + subindices +  '}$'
        indexes.append(_cfg_str)
        filename = f"{BRUTEFORCE_PERCENTAGE_FILENAME}_{'_'.join(map(str, _cfg))}_{db_size}.txt"
        plot_files.append(filename)
        total_time = sum(parse_values(algo, filename, 0))
        hashes_per_second = db_size / total_time
        times.append(hashes_per_second)
    
    indexes.append("Bruteforce")
    bruteforce_value = parse_bruteforce_values(algo, f"{BRUTEFORCE_FILENAME}.txt")
    times.append(bruteforce_value)
    df = pd.DataFrame({'Configurations': indexes, 'Hashes/s': times})
    create_barplot(algo, df, f"{BRUTEFORCE_PERCENTAGE_FILENAME}_{algo}", db_size, "Hashes/s", "Configurations", "Dataset with " + str(db_size) + " items")

def parse_values(algo, filename, values_position):
    data = []
    with open(f"{BENCHMARKS_INPUT_FOLDER}/{algo}/{filename}") as file:
        for line in file:
            value = float(line.split(': ')[values_position].replace('[','').replace(']',''))
            data.append(value)
    return data

def parse_knn_values(algo, filename, knn):
    data = []
    with open(f"{BENCHMARKS_INPUT_FOLDER}/{algo}/{filename}") as file:
        for line in file:
            numbers = eval(line.split(': ')[1])[:knn]
            data = data + numbers
    return data

def parse_percentage_values(algo, filename):
    data = []

    with open(f"{BENCHMARKS_INPUT_FOLDER}/{algo}/{filename}") as file:
        for line in file:
            numbers = eval(line.split(': ')[1])
            data.append(numbers)
    return data

def parse_bruteforce_values(algo, filename):
    with open(f"{BENCHMARKS_INPUT_FOLDER}/{algo}/{filename}") as file:
        value = float(file.read())
    return value

def create_boxplot(algo, df, plot_type, file_prefix, db_size, x_label, y_label, title):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, palette="Paired", orient=plot_type)
    _plot_str = "boxplot" + plot_type
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_facecolor('white')

    plt.title(title)
    plt.savefig(os.path.join(os.path.join(GRAPH_DIR, algo), file_prefix + f"_{db_size}_{_plot_str}.png"), bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def create_group_boxplot(algo, df, plot_type, file_prefix, db_size, x_label, y_label, title):
    fig, ax = plt.subplots()
    df = pd.melt(df, "configs")
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    sns.boxplot(data=df, hue="configs", x="variable", y="value", orient=plot_type)
    _plot_str = "boxplot" + plot_type
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_facecolor('white')

    plt.title(title)
    plt.legend(title='Configurations')
    plt.savefig(os.path.join(os.path.join(GRAPH_DIR, algo), file_prefix + f"_{db_size}_{_plot_str}.pdf"), bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def create_scatterplot(algo, df, plot_type, file_prefix, db_size, x_label, y_label):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='variable', y='value',
                     style='Configuration', hue='Configuration', palette='dark')
    _plot_str = "boxplot" + plot_type
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_facecolor('white')

    plt.savefig(os.path.join(os.path.join(GRAPH_DIR, algo), file_prefix + f"_{db_size}_{_plot_str}.pdf"), bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def create_barplot(algo, df, file_prefix, db_size, x_label, y_label, title):
    f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(6, 4))
    f.subplots_adjust(wspace=0.05)

    ax1 = sns.barplot(x=x_label, y=y_label, data=df, ax=ax1)
    ax2 = sns.barplot(x=x_label, y=y_label, data=df, ax=ax2)
    
    ax1.set_xlim(0, 30)
    ax2.set_xlim(100, 5000)

    ax2.get_yaxis().set_visible(False)

    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    ax1.bar_label(ax1.containers[0], fmt='%u')
    ax2.bar_label(ax2.containers[0], fmt='%u')
    
    d = 0.01
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d, +d), (-d, +d), **kwargs)  
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)    

    plt.savefig(os.path.join(os.path.join(GRAPH_DIR, algo), file_prefix + f"_{db_size}.pdf"), bbox_inches='tight')
    plt.clf()
    plt.close(f)

def add_labels(x, y, ax1, ax2):
    for i in range(len(x)):
        if i == len(x) - 1: # Bruteforce
            ax1.text(y[i] - 4, i, int(y[i]), ha = 'center')
        else: # CFGs
            ax2.text(y[i] - 500, i, int(y[i]), ha = 'center')

def clean():
    for algo in ALGORITHM:
        try:
            shutil.rmtree(os.path.join(GRAPH_DIR, algo))
        except:
            continue
        finally:
            os.mkdir(os.path.join(GRAPH_DIR, algo))
        
def create_plots():
    for algo in ALGORITHM:
        if algo == "ssdeep":
            DB_SIZES = [84136, 280455, 701138, 1402277, 2103416, 2804555]
        elif algo == "tlsh":
            DB_SIZES = [82533, 275110, 687777, 1375554, 2063331, 2751108]
        
        create_insertion_times_plot_summary(algo, DB_SIZES)
        create_search_times_plot_summary(algo, DB_SIZES)
        create_search_precision_plot_summary(algo, DB_SIZES)
    
        for size in DB_SIZES:
            create_insertion_times_plot(algo, size) # Insertion
            create_search_precision_plot(algo, size, FILE_TIMES_VALUES_POSITION) # KNN Seach times
            create_search_precision_plot(algo, size, FILE_PRECISION_VALUES_POSITION) # KNN Search precision
            create_precision_knn_plot(algo, size) # KNN Precision 1 to 10
            create_precision_percentage_plot(algo, size) # Percentage search precision


        create_bruteforce_vs_plot(algo, DB_SIZES[0])

    create_individual_parameter_plot_summary(algo, EF_PARAM_CFGs, "ef", [8,16,32,64])
    create_individual_parameter_plot_summary(algo, M_PARAM_CFGs, "M", [4,8,16,32])
    create_individual_parameter_plot_summary(algo, MMAX_PARAM_CFGs, "Mmax", [8,16,32,64])
    create_individual_parameter_plot_summary(algo, MMAX0_PARAM_CFGs, "Mmax0", [8,16,32,64])
    create_individual_parameter_precision_plot_summary(algo, EF_PARAM_CFGs, "ef", [8,16,32,64])
    create_individual_parameter_precision_plot_summary(algo, M_PARAM_CFGs, "M", [4,8,16,32])
    create_individual_parameter_precision_plot_summary(algo, MMAX_PARAM_CFGs, "Mmax", [8,16,32,64])
    create_individual_parameter_precision_plot_summary(algo, MMAX0_PARAM_CFGs, "Mmax0", [8,16,32,64])

if __name__ == "__main__":
    clean()
    create_plots()

