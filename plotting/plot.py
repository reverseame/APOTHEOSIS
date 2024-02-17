import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys
import seaborn as sns
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import matplotlib
from matplotlib import rc
import os
import shutil

rc('font',**{'family':'serif','serif':['Times'], 'size':10})
matplotlib.rcParams['text.usetex'] = True

script_name = os.path.basename(__file__)
PLOT_DIR = 'plots'
output_folder = os.path.join(os.getcwd(), PLOT_DIR)
LOG_DIR = 'logs'

df = None
X = 'MMAX'
Y = 'MMAX0'
Z = 'TIME'

#TODO
# Calcular correlación y covarianza entre variables?
# https://towardsdatascience.com/how-to-measure-relationship-between-variables-d0606df27fd8

def model(x1, y1):
    global X
    global Y
    global Z 
    global df_insert
    global df

    df_aux = df[df[Y] == y1]
    df_aux = df_aux[df_aux[X] == x1]
    return df_aux[Z].values[0]

def plot(df: pd.DataFrame, M: int, ef:int, npages: int, op: str, lims: tuple, hexa: bool=False) -> (str, str):
    global X
    global Y
    global Z 
    
    fig = plt.figure()
    threedee = fig.add_subplot(projection='3d')
    threedee.scatter(df[X], df[Y], df[Z],marker='*')
    threedee.set_xlabel(X)
    threedee.set_ylabel(Y)
    threedee.set_zlabel(Z)
    threedee.zaxis.set_major_formatter(FormatStrFormatter('%1.1e')) 
    zmin, zmax = lims
    threedee.set_zlim(zmin, zmax)

    xticks = range(df[X].min(), df[X].max() + 4, 4)
    threedee.set_xticks(xticks)
    yticks = range(df[Y].min(), df[Y].max() + 4, 4)
    threedee.set_yticks(yticks)

    if op == "INSERT":
        _str = "Insert time (ms)"
    else:
        _str = "Search time (ms)"
    threedee.set_zlabel(_str, rotation=90)

    #plt.show()
    filename = f"N{npages}_M{M}_X{X}_Y{Y}_Z{Z}_{op}"
    plt.title(f"{op} results for M={M}, ef={ef}, N={npages}")
    plt.savefig(os.path.join(output_folder, filename + "_scatter.pdf"), format="pdf")
    str_scatter = filename + "_scatter.pdf"
    plt.clf()

    x = df[X]
    y = df[Y]
    z = df[Z]

    if hexa:
        from sensitivity import SensitivityAnalyzer
        _dict = {'x1': x.values.tolist(), 'y1': y.values.tolist()}
        sa = SensitivityAnalyzer(_dict, model)
        df.plot.hexbin(x=X, y=Y, C=Z, gridsize=5, cmap=cm.inferno, sharex=False, xticks=xticks, yticks=yticks)
        plt.title(f"{op} results for M={M}, ef={ef}, N={npages}")
        plt.savefig(os.path.join(output_folder, filename + "_hex.pdf"), format="pdf")
    
    str_hex = filename + "_hex.pdf"
    
    plt.clf()
    plt.close()
    return str_scatter, str_hex


def print3D(df, order: int=2):
    global X
    global Y

    x = np.linspace(df[X].min(), df[X].max(), len(df[X].unique()))
    y = np.linspace(df[Y].min(), df[Y].max(), len(df[Y].unique()))
    XXX, YYY = np.meshgrid(x, y)

    # 1=linear, 2=quadratic, 3=cubic, ..., nth degree
    #order = 11

    # calculate exponents of design matrix
    #e = [(x,y) for x in range(0,order+1) for y in range(0,order-x+1)]
    e = [(x,y) for n in range(0,order+1) for y in range(0,n+1) for x in range(0,n+1) if x+y==n]
    eX = np.asarray([[x] for x,_ in e]).T
    eY = np.asarray([[y] for _,y in e]).T

    # best-fit polynomial surface
    A = (XXX ** eX) * (YYY ** eY)
    C,resid,_,_ = lstsq(A, Z)    # coefficients

    # calculate R-squared from residual error
    r2 = 1 - resid[0] / (Z.size * Z.var())

    # print summary
    print(f'data = {Z.size}x3')
    print(f'model = {exp2model(e)}')
    print(f'coefficients =\n{C}')
    print(f'R2 = {r2}')

    # uniform grid covering the domain of the data
    XX,YY = np.meshgrid(np.linspace(XXX.min(), XXX.max(), 20), np.linspace(YYY.min(), YYY.max(), 20))

    # evaluate model on grid
    A = (XX.reshape(-1,1) ** eX) * (YY.reshape(-1,1) ** eY)
    ZZ = np.dot(A, C).reshape(XX.shape)

    # plot points and fitted surface
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(XXX, YYY, Z, c='r', s=2)
    ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, alpha=0.2, linewidth=0.5, edgecolor='b')
    ax.axis('tight')
    ax.view_init(azim=-60.0, elev=30.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #plt.show()
    plt.savefig("prueba.pdf", format="pdf")


def create_wd():
    #if os.path.exists(output_folder):
    #    shutil.rmtree(output_folder)
    os.mkdir(output_folder)


def write_plots_latex(filename, insert, search, ncols):
    with open(filename, "w") as f:
        f.write("\\begin{longtable}{" + "c"*ncols + "}\n")
        f.write(insert.replace("0.5", str(1/ncols)))
        f.write("\\end{longtable}\n")
        f.write("\\newpage\n")
        f.write("\\begin{longtable}{" + "c"*ncols + "}\n")
        f.write(search.replace("0.5", str(1/ncols)))
        f.write("\\end{longtable}\n")
        f.close()

"""
breakpoint()
x = np.linspace(df[X].min(), df[X].max(), len(df[X].unique()))
y = np.linspace(df[Y].min(), df[Y].max(), len(df[Y].unique()))
x, y = np.meshgrid(x, y)
z = griddata((df[X], df[Z]), df[Z], (x, y), method='cubic')

breakpoint()



x = np.linspace(df[X].min(), df[X].max(), len(df[X].unique()))
y = np.linspace(df[Y].min(), df[Y].max(), len(df[Y].unique()))
x, y = np.meshgrid(x, y)
z = griddata((df[X], df[Z]), df[Z], (x, y), method='cubic')


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.jet, linewidth=0.2)
plt.savefig("".join(sys.argv[1].split('.')[:-1]) + ".pdf", format="pdf")

x, y = np.meshgrid(x, y)
for i in range(0, len(x) - 1):
    z = np.vstack([z, df[Z]])

plt.clf()
exit()

# re-create the 2D-arrays
x = np.linspace(df[X].min(), df[X].max(), len(df[X].unique()))
y = np.linspace(df[Y].min(), df[Y].max(), len(df[Y].unique()))
x, y = np.meshgrid(x, y)
z = griddata((df[X], df[Y]), df[Z], (x, y), method='cubic')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Meshgrid Created from 3 1D Arrays')

breakpoint()
#plt.plot(df.SEARCH, df.INSERT,'o',markersize=2, color='brown')
plt.savefig("".join(sys.argv[1].split('.')[:-1]) + ".pdf", format="pdf")
plt.clf()
"""

def get_graphs(df_csv: pd.DataFrame, ef, npages,\
                    zmin_insert, zmax_insert, zmin_search, zmax_search, hexa: bool=False):
    global df
    
    df_search = df_csv[df_csv["TYPE"] == "S"]
    df_insert = df_csv[df_csv["TYPE"] == "I"]

    insert_scatter = {}
    insert_hex = {}
    search_scatter = {}
    search_hex = {}
    for M in df_csv.M.unique():
        insert_scatter[M] = None
        search_scatter[M] = None
        insert_hex[M] = None
        search_hex[M] = None

        df = df_insert[df_insert["M"] == M]
        zmin_lim = zmin_insert 
        zmax_lim = zmax_insert
 
        str1, str2 = plot(df, M, ef, npages, "INSERT", (zmin_lim, zmax_lim), hexa)
        str_insert_scatter  = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str1) + "}"
        str_insert_hex      = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str2) + "}"
        insert_scatter[M] = str_insert_scatter
        insert_hex[M] = str_insert_hex
        
        df = df_search[df_search["M"] == M]
        zmin_lim = zmin_search
        zmax_lim = zmax_search
    
        str1, str2 = plot(df, M, ef, npages, "SEARCH", (zmin_lim, zmax_lim), hexa)
        str_search_scatter  = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str1) + "}"
        str_search_hex      = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str2) + "}"
        search_scatter[M] = str_search_scatter
        search_hex[M] = str_search_hex

    return insert_scatter, search_scatter, insert_hex, search_hex

if __name__ == "__main__":

    _dir = LOG_DIR
    if len(sys.argv) > 1:
        _dir = sys.argv[1]
    files = [f for f in os.listdir(_dir)]

    df_total = pd.DataFrame()
    for f in files:
        # we assume as filename: <arbitrary str>_<factor>_<ef>_<npages>.<ext>
        filename = "".join(f.split('.')[:-1])
        npages = filename.split('_')[-1]
        ef = filename.split('_')[-2] 
        
        df_csv = pd.read_csv(os.path.join(_dir, f))
        df_csv['ef']= ef
        df_csv['N']= npages
        df_total = pd.concat([df_total, df_csv])

    try: 
        create_wd()
    except:
        pass

    # z limits
    zmin_insert = df_total[df_total["TYPE"] == "I"][Z].min()
    zmax_insert = df_total[df_total["TYPE"] == "I"][Z].max()
    zmin_search = df_total[df_total["TYPE"] == "S"][Z].min()
    zmax_search = df_total[df_total["TYPE"] == "S"][Z].max()

    results = {}
    for N in sorted([int(n) for n in df_total.N.unique()]):
        results[N] = {}
        results[N]["I"] = {}
        results[N]["S"] = {}
        results[N]["I"]["scatter"] = {}
        results[N]["S"]["scatter"] = {} 
        results[N]["I"]["hex"] = {}
        results[N]["S"]["hex"] = {}

        df_N = df_total[df_total["N"] == str(N)]
        insert_scatter, search_scatter, insert_hex, search_hex = get_graphs(df_N, ef, str(N),\
                                                zmin_insert, zmax_insert, zmin_search, zmax_search)
        
        for k, v in insert_scatter.items():
            if results[N]["I"]["scatter"].get(k) is None:
               results[N]["I"]["scatter"][k] = None 
               results[N]["I"]["hex"][k] = None
               results[N]["S"]["scatter"][k] = None
               results[N]["S"]["hex"][k] = None

            results[N]["I"]["scatter"][k] = v
            results[N]["I"]["hex"][k] = insert_hex[k]
            results[N]["S"]["scatter"][k] = search_scatter[k]
            results[N]["S"]["hex"][k] = search_hex[k]

    str_insert_scatter = ""
    str_search_scatter = ""
    vM = sorted([int(n) for n in df_total.M.unique()])
    vN = sorted([int(n) for n in df_total.N.unique()])

    idx = 0
    for M in vM:
        for N in vN:
            str_insert_scatter += results[N]["I"]["scatter"][M] + " & "
            str_search_scatter += results[N]["S"]["scatter"][M] + " & "
            idx += 1
            if idx % (len(vN)/2) == 0:
                str_insert_scatter = str_insert_scatter[:-3] + "\\\\\n"
                str_search_scatter = str_search_scatter[:-3] + "\\\\\n"

    write_plots_latex(os.path.join("plotting", "plots.tex"), str_insert_scatter, str_search_scatter, int(len(vN)/2))
    
    str_insert_scatter = ""
    str_search_scatter = ""
    for N in vN:
        idx = 0
        for M in vM:
            str_insert_scatter += results[N]["I"]["scatter"][M] + " & "
            str_search_scatter += results[N]["S"]["scatter"][M] + " & "

            idx += 1
            if idx % (len(vM)/2) == 0:
                str_insert_scatter = str_insert_scatter[:-3] + "\\\\\n"
                str_search_scatter = str_search_scatter[:-3] + "\\\\\n"
    
    write_plots_latex(os.path.join("plotting", "plots2.tex"), str_insert_scatter, str_search_scatter, int(len(vM)/2))


