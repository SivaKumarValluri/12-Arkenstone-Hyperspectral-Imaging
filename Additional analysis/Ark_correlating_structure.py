"""
Created on Tue Apr  1 10:45:47 2025

@author: Siva Kumar Valluri
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression  

data = {
     #'BPR': [3, 5, 3, 3, 5, 5],
     #'RPM': [250, 250, 350, 350, 350, 350],
     #'PCA vol': [24, 14.5, 24, 20, 14.5, 12],
     #'m/V': [1.25, 1.241379, 1.25, 1.5, 1.241379, 1.5],
     #'powder charge': [30, 18, 30, 30, 18, 18],
     #'milling dose': [0, 0.22437, 0.32877, 0.46301, 0.76854, 1],
     #'mixing granularity': [5.2, 4.7, 4.7, 4.5, 4.4, 4.3],
     #'CuO perimeter/area': [12.42251, 20.06812, 29.25945, 31.40716, 27.15723, 26.33352],
     #'pore diameter': [0.11373, 0.27315, 0.19602, 0.14067, 0.10094, 0.11868],
     #'Cu atomic%': [7.98, 6.87, 4.87, 4.67, 3.67, 3.33],
     'endotherm': [0, 0, 25.47, 38.7, 81.58, 78.51],                   # J/g from DSC
     'exotherm': [1011, 1671, 1082, 1158, 778.8, 674.2],               # J/g from DSC
     'ignition temperature': [1666.812, 1420.015, 1521.124, 1433.803, 1462.756, 1385.24],  # K
     'probability-4.0 km/s': [0.03111, 0.05941, 0.0453, 0.03784, 0.00962, 0.02381],        # shock exp
     'probability-4.5 km/s': [0.09202, 0.12201, 0.10048, 0.05714, 0.14386, 0.1576]         # shock exp

    #PBX detail (they're ordered differently note)
    #'probability-4.5 km/s':[0.09202,0.12201,0.10048,0.05714,0.14386,0.1576],
    #'milling dose':[234375, 387931.0345, 459375, 551250, 760344.8276, 918750],
    #'average pore diameter':[0.48118, 0.44409, 0.38308, 0.3424, 0.31773, 0.32],
    #'endotherm':[0,0,-25.47,-38.7,-81.58, -78.51],
    #'exotherm':[1011,1671,1082,1158,778.8,674.2],
    #'ignition temperature':[1666.81222,1420.01514,1521.12382,1433.80269,1462.7556,1385.23989],
    #'shock':[215160.2877,98395.43961,531503.3386,428333.4164,551364.6333,181619.329],
    #'particle':[2063104.69,1998384.251,2245263.822,2028077.218,1450768.404,1246290.012],
    #'bulk':[4476407.27,4595104.83,6074411.318,7482984.089,7413550.5,6837143.955]
    }
dataset = pd.DataFrame(data)

# Z-score normalization
from scipy.stats import zscore
dataset_norm = dataset.apply(zscore)

#Correlation (Pearson, Spearman and Kendall)
def plot_correlation_heatmap(dataframe, title, method=None):
    """
    Plots correlation heatmaps for the given dataframe.
    Also prints strong positive and negative correlations.
    
    Parameters:
    - dataframe: pd.DataFrame containing the data to correlate
    - title: str, the base title for the heatmap plots
    - method: str or None, correlation method ('pearson', 'spearman', 'kendall')
              If None, plots all three.
    """
    methods_to_plot = [method] if method else ['pearson', 'spearman', 'kendall']

    # Recommended thresholds
    thresholds = {
        'pearson': 0.7,
        'spearman': 0.7,
        'kendall': 0.5
    }

    # Set global font
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20

    for m in methods_to_plot:
        correlation_matrix = dataframe.corr(method=m)
        threshold = thresholds[m]

        # Plot heatmap
        plt.figure(figsize=(8, 7))
        ax = sns.heatmap(
            correlation_matrix,
            cmap=sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True),
            square=True,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8, "label": f"{m.capitalize()} correlation coefficient"}
        )

        # Set colorbar font
        colorbar = ax.collections[0].colorbar
        colorbar.ax.yaxis.label.set_font_properties(
            plt.matplotlib.font_manager.FontProperties(family='Arial', size=16)
        )

        plt.title(f"{title} ({m.capitalize()})")
        plt.show()

        # Print strong correlations
        print(f"\nFor method '{m.capitalize()}':")
        for col in correlation_matrix.columns:
            strong_pos = []
            strong_neg = []
            for other_col in correlation_matrix.columns:
                if col != other_col:
                    val = correlation_matrix.loc[col, other_col]
                    if val > threshold:
                        strong_pos.append(other_col)
                    elif val < -threshold:
                        strong_neg.append(other_col)

            if strong_pos:
                print(f"'{col}' is strongly positively correlated with: " + ', '.join([f"'{x}'" for x in strong_pos]))
            if strong_neg:
                print(f"'{col}' is strongly negatively correlated with: " + ', '.join([f"'{x}'" for x in strong_neg]))
            if not strong_pos and not strong_neg:
                print(f"'{col}' has no strong correlations.")

"""
selected_columns = ['endotherm',
    'exotherm',
    'ignition temperature',
    'average CuO shape factor',
    'average pore diameter',
    'milling dose',
    'probability-4.0 km/s',
    'probability-4.5 km/s' 
]
"""
selected_columns = dataset_norm.columns    
new_df = dataset_norm[selected_columns]
plot_correlation_heatmap(new_df, "Relationship between structural features, preliminary tests and shock-initiated behavior")


# Feature Selection with ANOVA (Regression version)
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

def select_features_anova(dataset, target_column):
    # Ensure target_column exists
    if target_column not in dataset.columns:
        raise ValueError(f"'{target_column}' not found in dataset columns.")

    # Split into features (X) and target (y)
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Apply SelectKBest with f_regression
    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X, y)

    # Extract scores and p-values
    scores = selector.scores_
    p_values = selector.pvalues_

    # Prepare results DataFrame
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'F-Score': scores,
        'P-Value': p_values
    }).sort_values(by='F-Score', ascending=False)

    # Selected features (those retained by selector)
    selected_features = X.columns[selector.get_support()].tolist()

    print(f"\nTarget column: '{target_column}'")
    print("Selected features by ANOVA (f_regression):", selected_features)
    print("\nFeature scores:")
    print(feature_scores.to_string(index=False))

    return selected_features, feature_scores


#target = 'probability-4.5 km/s'  # user-defined
#selected, scores_df = select_features_anova(dataset, target)

# Feature selection MANOVA
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

def run_manova(dataset, dependent_vars, independent_vars):
    """
    Perform MANOVA on given dataset.

    Parameters:
    - dataset: pd.DataFrame
    - dependent_vars: list of column names to treat as dependent variables (Y)
    - independent_vars: list of column names to treat as independent variables (X)

    Returns:
    - manova_result: statsmodels MANOVA object (printable summary)
    """

    # Build formula: 'Y1 + Y2 + ... ~ X1 + X2 + ...'
    dep_str = ' + '.join(dependent_vars)
    indep_str = ' + '.join(independent_vars)
    formula = f'{dep_str} ~ {indep_str}'

    # Run MANOVA
    manova = MANOVA.from_formula(formula, data=dataset)
    result = manova.mv_test()
    
    print("MANOVA formula:", formula)
    return result


dependent_cols = ['bulk']
independent_cols = ['mixing granularity', 'CuO perimeter/area', 'pore diameter',
       'Cu atomic%', 'endotherm', 'exotherm', 'ignition temperature',
       'probability-4.0 km/s']


manova_output = run_manova(dataset, dependent_cols, independent_cols)
print(manova_output)


# PCA Analysis
target_column = 'probability-4.0 km/s'
feature_columns = dataset.columns.drop(target_column)

# Define normalization methods
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler()
}

# Apply PCA for each scaler
for scaler_name, scaler in scalers.items():
    # Scale features
    X = dataset[feature_columns]
    y = dataset[target_column]
    X_scaled = scaler.fit_transform(X)

    # PCA to retain 95% variance
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Full PCA for loadings and biplot
    pca_full = PCA().fit(X_scaled)
    loadings = pca_full.components_.T * np.sqrt(pca_full.explained_variance_)

    # Set fonts
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 14})

    # Scree Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
             pca_full.explained_variance_ratio_, marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title(f'Scree Plot ({scaler_name})\nTarget: {target_column}')
    plt.grid(True)
    plt.show()

    # Biplot for PC1 vs PC2
    pc_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    pc_df[target_column] = y

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='PC1', y='PC2', hue=target_column, data=pc_df, palette='coolwarm', s=100)

    for i, feature in enumerate(feature_columns):
        x = loadings[i, 0]
        y_arrow = loadings[i, 1]
        plt.arrow(0, 0, x, y_arrow, color='black', alpha=0.6, head_width=0.03)
        plt.text(x * 1.2, y_arrow * 1.2, feature, fontsize=14, ha='center', va='center', fontweight='bold')

    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)
    plt.title(f'PCA Biplot (PC1 vs PC2) - {scaler_name}\nTarget: {target_column}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.text(1.5, -1.2, "➡ Long arrow = important feature\n➡ Similar arrow direction = correlated features\n➡ Points near arrows = high values for that feature",
             fontsize=12, bbox=dict(facecolor='white', alpha=0.6), ha='left')
    plt.show()

    # Loadings summary
    top_features = {
        f"PC{i+1}": feature_columns[np.argmax(np.abs(row))]
        for i, row in enumerate(pca_full.components_)
    }

    loadings_df = pd.DataFrame(
        data=np.round(pca_full.components_, 4),
        columns=feature_columns,
        index=[f"PC{i+1}" for i in range(len(pca_full.components_))]
    )

    print(f"\nMost important feature in each principal component ({scaler_name}):")
    for pc, feat in top_features.items():
        print(f"{pc}: {feat}")
    print(f"\nPCA Loadings Table ({scaler_name}):\n")
    print(loadings_df)
    print("\n" + "="*80 + "\n")

