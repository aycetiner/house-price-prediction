# Exploratory Data Analysis Functions

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def check_df(dataframe, head=5):
    """
    Returns information of the given dataframe.

    Parameters
    ----------
    dataframe: The dataframe you want to check
    head: The number of first and last observations you want to observe

    Returns
    -------
    shape of df
    datatype of each column
    first 'head' rows of the df (default is 5)
    last 'head' rows of the df (default is 5)
    missing value count for each column
    quantiles
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    """
    Returns a summary for categorical columns.

    Parameters
    ----------
    dataframe
    col_name
    plot

    Returns
    -------

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def cat_distribution_plot(df, cat_col, target_col):
    """
    Returns a histogram of the observation unit distribution of target attribute to categorical attributes.

    Parameters
    ----------
    df: Dataframe
    cat_col: Categorical column names
    target_col: Target column name

    Returns
    -------

    """
    split_hist(df, target_col, split_by=cat_col, ylabel="Observation Count",
               title=f"{target_col}'s distribution over {cat_col}", bins=25, figsize=(12.1,4))
    plt.show()


def cat_analyser_plot(df, cat_col, target_col):
    """
    Returns the plot for value count distribution for categorical columns and the plot for target mean distribution
    for categorical columns.

    Parameters
    ----------
    df
    cat_col
    target_col

    Returns
    -------

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df[cat_col].value_counts().plot(kind='bar', ax=axes[0], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].set_title(f"{cat_col} Distribution")
    axes[0].set_xlabel(f"{cat_col}")
    axes[0].set_ylabel("Observation Count")

    df.groupby(cat_col)[target_col].mean().plot(kind='bar', ax=axes[1],
                                                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1].set_title(f"{target_col}" + " Mean Distribution Over " + f"{cat_col}")
    axes[1].set_xlabel(f"{cat_col}")
    axes[1].set_ylabel(f"{target_col}")
    plt.xticks(rotation=0)
    plt.show()

    cat_distribution_plot(df, cat_col, target_col)


def num_summary(dataframe, numerical_col, plot=False):
    """
    Returns Numerical Variable Summary Statistics in histogram plot.

    Parameters
    ----------
    dataframe
    numerical_col
    plot

    Returns
    -------

    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print("\t\t\t"+f"{numerical_col}"+" Numerical Variable Summary Statistics")
    print("\t\t\t---------------------------------------------")
    print(pd.DataFrame(dataframe[numerical_col].describe(quantiles)).T, "\n")

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


def num_analyser_plot(df, num_col, target_col):
    """
    Returns the plot for value count distribution for numerical columns and the plot for target mean distribution
    for numerical columns.

    Parameters
    ----------
    df
    num_col
    target_col

    Returns
    -------

    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[num_col], kde=True, bins=30, ax=axes[0]);
    axes[0].lines[0].set_color('green')
    axes[0].set_title(f"{num_col}" + " " + "Distribution")
    axes[0].set_ylabel("Observation Count")

    quantiles = [0, 0.25, 0.50, 0.75, 1]
    num_df = df.copy()
    num_df[f"{num_col}" + "_CAT"] = pd.qcut(df[num_col], q=quantiles)  # numerical variables are categorized.
    df_2 = num_df.groupby(f"{num_col}" + "_CAT")[target_col].mean()

    sns.barplot(x=df_2.index, y=df_2.values);
    axes[1].set_title(f"{target_col} Mean Over {num_col}")
    axes[1].set_ylabel(f"{target_col}")

    plt.show()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = variable count.
    # num_but_cat is in cat_cols.
    # 3 lists to cover every variable: cat_cols + num_cols + cat_but_car
    # num_but_cat is given extra for reporting purposes.

    return cat_cols, cat_but_car, num_cols, num_but_cat


def target_summary_with_cat(dataframe, target, categorical_col):
    """
    A snapshot of target variable summary over categorical variables.
    Parameters
    ----------
    dataframe
    target
    categorical_col

    Returns
    -------

    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean().sort_values(ascending=False)}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    """
        A snapshot of target variable summary over numerical variables
        Parameters
        ----------
        dataframe
        target
        categorical_col

        Returns
        -------

        """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")



def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    Correlation matrix for the variables of given dataframe.

    Parameters
    ----------
    dataframe
    plot
    corr_th

    Returns
    -------

    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def outliers_boxplot(dataframe, num_cols):
    """
    Outlier Analysis of Numerical Variables with boxplot.
    Parameters
    ----------
    dataframe
    num_cols

    Returns
    -------
    Returns a boxplot for numerical variables, good for observing outliers.

    """
    plt.figure(figsize=(12,6),dpi=200)
    plt.title("Outlier Analysis of Numerical Variables with Boxplot")
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=df.loc[:, num_cols], orient="h", palette="Set3")
    plt.show()

def create_ratio_cols(dataframe, numerator_col, denominator_col, new_col_name=False):
    """
    Creates new ratio column from other numerical columns.
    Parameters
    ----------
    dataframe
    numerator_col
    denominator_col
    new_col_name

    Returns
    -------

    """
    if new_col_name:  # if the new column is given a name by the function argument new_col_name
        dataframe[new_col_name] = dataframe[numerator_col]/(dataframe[denominator_col]+0.001)
    else:              # 0.001 is added to prevent having zero in denominator.
        dataframe[f"NEW_{numerator_col}/{denominator_col}"] = dataframe[numerator_col]/(dataframe[denominator_col]+0.0001)
