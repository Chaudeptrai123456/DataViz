import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def cali():
    scatter()
def seaborn():
    df_tips = sns.load_dataset('tips')
    # count_total_bill = df_tips['total_bill'].value_counts(ascending=False)
    # print(count_total_bill)

    #draw displot
    # sns.histplot(data = df_tips["total_bill"])
    #sns.kdeplot(data = df_tips["total_bill"])
    # sns.displot(data = df_tips, x = "total_bill", col = 'day',kde=True )
    
    #barplot
    # sns.barplot(df_tips,x="sex",y="tip",estimator = np.mean)
    
    # countplot
    # sns.countplot(df_tips,x="day")

    #boxplot
    sns.boxplot(df_tips,x="day",y="tip",hue="sex")
    
    plt.show()
    print(df_tips.head())
def scatter():
    cities = pd.read_csv(".\california_cities.csv")
    latd ,longd = cities["latd"],cities["longd"]
    population, area = cities["population_total"],cities["area_total_km2"]
    plt.figure(figsize=(10,10))
    # plt.style.use("seaborn")
    plt.scatter(latd ,longd,c = population/10000,s=area)
    area_range = [50,100,150,200]
    for area in area_range:
        plt.scatter([],[],s=area,label= str(area)  + " km$^2$")
    plt.legend(title="City area")
    # plt.colorbar(label ="log$_{10}${population}")
    plt.colorbar(label ="10000 citizen")
    plt.show()
def iris_mean():
    df_iris = pd.read_csv("https://raw.githubusercontent.com/phamdinhkhanh/datasets/master/iris_train.csv", header=0, index_col=None)
    df_mean_sepal = df_iris[['Species', 'Petal.Length']].groupby("Species")
    y = df_mean_sepal.mean().values
    x=  list(df_mean_sepal.mean().index)
    y1 = y.reshape(3,)
    drawBar(x,y1,"Mean")
def histogram():
    a = np.random.normal(50,3,100)
    fig,ax = plt.subplots()
    plt.hist(a,bins=30)
    plt.show()
def drawscatter():
    rgn= np.random.RandomState(0)
    x = rgn.randn(100)
    y = rgn.randn(100)
    size = 1000*rgn.rand(100)
    color = rgn.rand(100)
    fig, ax = plt.subplots()
    img =  ax.scatter(x,y,s=size,c=color,alpha=0.5)
    fig.colorbar(img) 
    plt.show()
def drawline():
    print("line")
def drawLine(x,y,title):
    print(title)
def drawBar(x,y,title):
    plt.title(title)
    plt.bar(x,y)
    plt.xlabel('Species', fontsize=16)
    plt.ylabel('cm', fontsize=16)
    plt.title("Average of Petal Length", fontsize=18)
    plt.show()

# def draw():
#     x = np.linspace(0,10,1000)
#     plt.title("Sin and Cos")
#     plt.xlabel("x")
#     plt.ylabel("Sin(x)")
#     # plt.xlim([0,1])
#     # plt.ylim([-1,1])
#     # y=[1,2,3,4]
#     plt.plot(x,np.sin(x),color="red",linestyle="dashed",label="sin(x)")
#     plt.plot(x,np.cos(x),color="blue",label="cos(x)")
#     # hien ra toa do 
#     plt.legend()
#     plt.axis("tight")
#     plt.show()
# def drawOO():
#     x = np.linspace(0,10,1000)
#     fig,ax = plt.subplots(figsize=(5,5))
#     ax.plot(x,x**2)
#     ax.set( title="Test choi cho vui",
#             xlabel="x",
#             ylabel="Sin")
#     plt.show()

def _3d():
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

# fake data
    _x = np.arange(4)
    _y = np.arange(5)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = x + y
    bottom = np.zeros_like(top)
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Shaded')

    ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
    ax2.set_title('Not Shaded')

    plt.show()