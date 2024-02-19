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
import matplotlib.colors as colors
from matplotlib import rc
import os
import shutil

rc('font',**{'family':'serif','serif':['Times'], 'size':10})
matplotlib.rcParams['text.usetex'] = True

script_name = os.path.basename(__file__)
PLOT_DIR = 'plots'
output_folder = os.path.join(os.getcwd(), PLOT_DIR)
LOG_DIR = 'logs'
TEX_DIR = 'tex'

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
    
    zmin, zmax = lims
    fig = plt.figure()
    threedee = fig.add_subplot(projection='3d')
    threedee.scatter(df[X], df[Y], df[Z],marker='o', c=df[Z], cmap='winter',\
                    norm=colors.Normalize(vmin=zmin, vmax=zmax))
    threedee.set_xlabel(X)
    threedee.set_ylabel(Y)
    threedee.set_zlabel(Z)
    threedee.zaxis.set_major_formatter(FormatStrFormatter('%1.1e')) 
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
    try:
        os.mkdir(output_folder)
    except Exception as e:
        pass
    finally:
        os.mkdir(os.path.join("plotting", TEX_DIR))

def write_plots_latex(filename, insert, search_exact, search_approximate, ncols):
    with open(filename, "w") as f:
        f.write("\\begin{longtable}{" + "c"*ncols + "}\n")
        f.write(insert.replace("0.5", str(1/ncols)))
        f.write("\\end{longtable}\n")
        f.write("\\newpage\n")
        f.write("\\begin{longtable}{" + "c"*ncols + "}\n")
        f.write(search_exact.replace("0.5", str(1/ncols)))
        f.write("\\end{longtable}\n")
        f.write("\\newpage\n")
        f.write("\\begin{longtable}{" + "c"*ncols + "}\n")
        f.write(search_approximate.replace("0.5", str(1/ncols)))
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
                    zmin_insert, zmax_insert,\
                    zmin_search_exact, zmax_search_exact, zmin_search_approx, zmax_search_approx,\
                    hexa: bool=False):
    global df
    
    df_search_exact  = df_csv[df_csv["TYPE"] == "SE"]
    df_search_approx = df_csv[df_csv["TYPE"] == "SA"]
    df_insert        = df_csv[df_csv["TYPE"] == "I"]

    insert_scatter        = {}
    insert_hex            = {}
    search_exact_scatter  = {}
    search_exact_hex      = {}
    search_approx_scatter = {}
    search_approx_hex     = {}
    for M in df_csv.M.unique():
        insert_scatter[M]        = None
        search_exact_scatter[M]  = None
        search_approx_scatter[M] = None
        insert_hex[M]            = None
        search_exact_hex[M]      = None
        search_approx_hex[M]     = None

        df       = df_insert[df_insert["M"] == M]
        zmin_lim = zmin_insert 
        zmax_lim = zmax_insert
 
        str1, str2          = plot(df, M, ef, npages, "INSERT", (zmin_lim, zmax_lim), hexa)
        str_insert_scatter  = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str1) + "}"
        str_insert_hex      = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str2) + "}"
        insert_scatter[M]   = str_insert_scatter
        insert_hex[M]       = str_insert_hex
        
        df       = df_search_exact[df_search_exact["M"] == M]
        zmin_lim = zmin_search_exact
        zmax_lim = zmax_search_exact
    
        str1, str2                  = plot(df, M, ef, npages, "EXACT SEARCH", (zmin_lim, zmax_lim), hexa)
        str_search_exact_scatter    = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str1) + "}"
        str_search_exact_hex        = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str2) + "}"
        search_exact_scatter[M]     = str_search_exact_scatter
        search_exact_hex[M]         = str_search_exact_hex
        
        df       = df_search_approx[df_search_approx["M"] == M]
        zmin_lim = zmin_search_approx
        zmax_lim = zmax_search_approx
    
        str1, str2                  = plot(df, M, ef, npages, "AKNN SEARCH", (zmin_lim, zmax_lim), hexa)
        str_search_approx_scatter   = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str1) + "}"
        str_search_approx_hex       = "\\includegraphics[width=0.5\\columnwidth]{" + os.path.join("../", PLOT_DIR, str2) + "}"
        search_approx_scatter[M]    = str_search_approx_scatter

    return insert_scatter, search_exact_scatter, search_approx_scatter, insert_hex, search_exact_hex, search_approx_hex

def plot_M_N(df_total: pd.DataFrame, ef):
    # z limits
    zmin_insert = df_total[df_total["TYPE"] == "I"][Z].min()
    zmax_insert = df_total[df_total["TYPE"] == "I"][Z].max()
    zmin_search_exact   = df_total[df_total["TYPE"] == "SE"][Z].min()
    zmax_search_exact   = df_total[df_total["TYPE"] == "SE"][Z].max()
    zmin_search_approx  = df_total[df_total["TYPE"] == "SA"][Z].min()
    zmax_search_approx  = df_total[df_total["TYPE"] == "SA"][Z].max()

    results = {}
    for N in sorted([int(n) for n in df_total.N.unique()]):
        results[N] = {}
        results[N]["I"] = {}
        results[N]["SE"] = {}
        results[N]["SA"] = {}
        results[N]["I"]["scatter"] = {}
        results[N]["SE"]["scatter"] = {} 
        results[N]["SA"]["scatter"] = {} 
        results[N]["I"]["hex"] = {}
        results[N]["SE"]["hex"] = {}
        results[N]["SA"]["hex"] = {}

        df_N = df_total[df_total["N"] == str(N)]
        insert_scatter, search_exact_scatter, search_approx_scatter,\
            insert_hex, search_exact_hex, search_approx_hex \
                                        = get_graphs(df_N, ef, str(N),\
                                                zmin_insert, zmax_insert,\
                                                zmin_search_exact, zmax_search_exact, zmin_search_approx, zmax_search_approx)
        
        for k, v in insert_scatter.items():
            if results[N]["I"]["scatter"].get(k) is None:
               results[N]["I"]["scatter"][k] = None 
               results[N]["I"]["hex"][k] = None
               results[N]["SE"]["scatter"][k] = None
               results[N]["SE"]["hex"][k] = None
               results[N]["SA"]["scatter"][k] = None
               results[N]["SA"]["hex"][k] = None

            results[N]["I"]["scatter"][k]   = v
            results[N]["I"]["hex"][k]       = insert_hex[k]
            results[N]["SE"]["scatter"][k]  = search_exact_scatter[k]
            results[N]["SE"]["hex"][k]      = search_exact_hex[k]
            results[N]["SA"]["scatter"][k]  = search_approx_scatter[k]
            results[N]["SA"]["hex"][k]      = search_approx_hex[k]

    str_insert_scatter        = ""
    str_search_exact_scatter  = ""
    str_search_approx_scatter = ""
    vM = sorted([int(n) for n in df_total.M.unique()])
    vN = sorted([int(n) for n in df_total.N.unique()])

    idx = 0
    for M in vM:
        for N in vN:
            str_insert_scatter          += results[N]["I"]["scatter"][M]  + " & "
            str_search_exact_scatter    += results[N]["SE"]["scatter"][M] + " & "
            str_search_approx_scatter   += results[N]["SA"]["scatter"][M] + " & "
            idx += 1
            if idx % (len(vN)/2) == 0:
                str_insert_scatter        = str_insert_scatter[:-3]        + "\\\\\n"
                str_search_exact_scatter  = str_search_exact_scatter[:-3]  + "\\\\\n"
                str_search_approx_scatter = str_search_approx_scatter[:-3] + "\\\\\n"

    filename1 = f"plotsMvsN_ef{ef}.tex"
    write_plots_latex(os.path.join("plotting", TEX_DIR, filename1),\
                        str_insert_scatter, str_search_exact_scatter, str_search_approx_scatter,\
                        int(len(vN)/2) if len(vN) > 1 else 1)
    
    str_insert_scatter        = ""
    str_search_exact_scatter  = ""
    str_search_approx_scatter = ""
    for N in vN:
        idx = 0
        for M in vM:
            str_insert_scatter          += results[N]["I"]["scatter"][M]  + " & "
            str_search_exact_scatter    += results[N]["SE"]["scatter"][M] + " & "
            str_search_approx_scatter   += results[N]["SA"]["scatter"][M] + " & "

            idx += 1
            if idx % (len(vM)/2) == 0:
                str_insert_scatter = str_insert_scatter[:-3] + "\\\\\n"
                str_search_exact_scatter = str_search_exact_scatter[:-3] + "\\\\\n"
                str_search_approx_scatter = str_search_approx_scatter[:-3] + "\\\\\n"

    filename2 = f"plotsNvsM_ef{ef}.tex"
    write_plots_latex(os.path.join("plotting", TEX_DIR, filename2),\
                        str_insert_scatter, str_search_exact_scatter, str_search_approx_scatter,\
                        int(len(vM)/2) if len(vM) > 1 else 1)

    return filename1, filename2

if __name__ == "__main__":

    _dir = LOG_DIR
    if len(sys.argv) > 1:
        _dir = sys.argv[1]
    files = [f for f in os.listdir(_dir)]

    df_total = pd.DataFrame()
    for f in files:
        # we assume as filename: <arbitrary str>_<factor>_<npages>_<nsearch-pages>.<ext>
        filename = "".join(f.split('.')[:-1])
        npages = filename.split('_')[-2]
        nsearch_pages = filename.split('_')[-1]
        
        df_csv = pd.read_csv(os.path.join(_dir, f))
        df_csv['N']= npages
        df_csv['SEARCH-PAGES']= nsearch_pages
        df_total = pd.concat([df_total, df_csv])

    try: 
        create_wd()
    except:
        pass
    
    f = open(os.path.join("plotting", "plots.tex"), "w")
    vEF = sorted([int(n) for n in df_total.EF.unique()])
    for ef in vEF:
        str1, str2 = plot_M_N(df_total[df_total["EF"] == ef], ef)
        f.write("\\input{" + os.path.join(TEX_DIR, str1) + "}\n")
        f.write("\\newpage\n")
        f.write("\\input{" + os.path.join(TEX_DIR, str2) + "}\n")
    f.close()
