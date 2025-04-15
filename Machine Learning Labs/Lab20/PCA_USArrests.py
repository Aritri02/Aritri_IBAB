import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ISLP import load_data
from statsmodels.datasets import get_rdataset

def biplot(score, coeff, labels=None, states=None):
    xs = score[:, 0]
    ys = score[:, 1]
    zs = score[:, 2]
    n = coeff.shape[0]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, s=5)

    if states is not None:
        for i in range(len(xs)):
            ax.text(xs[i], ys[i], zs[i], states[i], size=7)

    for i in range(n):
        ax.quiver(0, 0, 0, coeff[i, 0], coeff[i, 1], coeff[i, 2], color='r', alpha=0.5)
        label = labels[i] if labels is not None else f"Var{i+1}"
        ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, coeff[i, 2] * 1.15, label, color='g', ha='center', va='center')

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("3D Biplot")
    plt.grid()
    plt.show()

def main():
    # Load data
    df = get_rdataset('USArrests').data
    # df = pd.read_csv("/home/ibab/Downloads/USArrests.csv", index_col=0)

    print(df.head())
    print(df.info())
    print(df.describe())

    ##############From the dataset summary above the following observations can be made#######
    # The dataset contains 50 rows and 5 columns.
    # The four variables have vastly different means
    # The variables also have vastly different variances
    # There are no null values to report in the dataset

    ##########Data visualization############
    # 1.Histograms:
    df.hist(grid=False, figsize=(10, 5))
    plt.tight_layout()
    plt.show()

    # 2.Plot of crime rate & urban population
    fig, ax = plt.subplots(figsize=(10,30))
    y = np.arange(len(df.axes[0]))  # the label locations
    bar_height = 0.4  # the height of the bars

    # set the position of the bars on the y-axis
    bar_positions = y - bar_height
    assult = ax.barh(bar_positions, df.Assault, bar_height, color='g')
    rape = ax.barh(bar_positions, df.Rape, bar_height, color='b', left=df.Assault)
    murder = ax.barh(bar_positions, df.Murder, bar_height, color='r', left=df.Assault+df.Rape)
    urbanpop = ax.barh(bar_positions + bar_height, df.UrbanPop, bar_height, color='cyan')
    ax.set_yticks(y)  # set the y-ticks to be at the same position as the bars
    ax.set_yticklabels(df.axes[0])  # set the y-tick labels to be the labels of the dataframe
    plt.legend(['Assault','Rape','Murder','UrbanPop'])
    plt.margins(y=0)
    plt.show()

    ## the following observations are made from the barchart:
    # Highest Assault Rate : Florida and North California.
    # Lowest Assault Rate : Hawaii, North Dakota, Vermont , New Hampshire and Wisconsin.
    # Highest Rape Rate : Nevada and Alaska.
    # Lowest Rape Rate : Maine, North Dakota,Vermont,Connecticut,New Hampshire, Wisconsin,Rhode Island and West Virginia
    # Highest Murder Rate : Georgia and Missisippi
    # Lowest Murder Rate : Idaho , Iowa, Maine, New Hampshire, North Dakota, Vermont and Wisconsin.
    # Highest UrbanPop Rate : Nevada and Alaska.
    # Lowest UrbanPop Rate : Maine, North Dakota,Vermont,Connecticut,New Hampshire, Wisconsin,Rhode Island and West Virginia

    ## 3.Correlation analysis
    states = df.index
    corr_df = df.corr()
    labels = corr_df.columns
    mask_ut = np.triu(np.ones(corr_df.shape)).astype(bool)
    sns.heatmap(corr_df, mask=mask_ut, annot=True, cmap="viridis")
    plt.show()

    ## the following are heat map observations:
    # Rate of arrests for assault has very strong positive correlation with the rate of arrests for murder.
    # Rate of arrests for assault has strong positive correlation with the rate of arrests for rape.
    # Rate of arrests for murder has moderate positive correlation with the rate of arrests for rape.
    # Urbanpopulation percentage has moderate positive correlation with the rate of arrests for rape.
    # Urbanpopulation percentage has weak positive correlation with the rate of arrests for assault.
    # Urbanpopulation percentage has almost no correlation with the rate of arrests for murder.

    ## 4.Pairplot
    sns.pairplot(df, kind='reg')
    plt.show()

    ## pair plot observations:
    # Murder , Assault & Rape are highly co-related to each other.
    # UrbanPop is not in co-relation with other variables.
    # Because of the highly co-related input data, PCA can be applied to reduce the number of features.

    X = df.values.squeeze()

    # data standardization
    X_std = StandardScaler().fit_transform(X)
    pca = PCA()
    X_std_trans = pca.fit_transform(X_std)
    df_std_pca = pd.DataFrame(X_std_trans)
    std = df_std_pca.describe().transpose()["std"]
    print(f"Standard deviation: {std.values}")
    print(f"Proportion of Variance Explained: {pca.explained_variance_ratio_}")
    print(f"Cumulative Proportion: {np.cumsum(pca.explained_variance_)}")

    # 3D biplot
    biplot(X_std_trans[:, 0:3], np.transpose(pca.components_[0:3, :]), labels=list(labels), states=states)

    ### From this biplot, we see that Assault and UrbanPop are the most important features as the arrows to each of these dominate the biplot.

    # Feature importance for the first 3 principal components
    pc1 = abs(pca.components_[0])
    pc2 = abs(pca.components_[1])
    pc3 = abs(pca.components_[2])
    feat_df = pd.DataFrame()
    feat_df["Features"] = list(labels)
    feat_df["PC1 Importance"] = pc1
    feat_df["PC2 Importance"] = pc2
    feat_df["PC3 Importance"] = pc3
    print(feat_df)

    # Inspecting the feature importance now, it seems that most of the variables contribute fairly evenly, with only some with low importance.

    # Cumulative variance plot
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_),
             c='red')
    plt.title("Cumulative Explained Variance")
    plt.show()

    # Scree plot
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title("Scree plot")
    plt.show()

    ## From the plots above, it seems the first 3 principal components together explain around 95% of the variance. We can therefore use them to perform model training.

    # PCA dataset creation:
    pca_df = pd.DataFrame(X_std_trans[:, 0:3], index=df.index)
    print("##########################################################")
    print(pca_df.head())

if __name__ == "__main__":
    main()
