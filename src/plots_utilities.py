"""
This module provides a collection of plotting and visualization functions
for exploratory data analysis (EDA) and machine learning model evaluation.

It includes functions for:

- Visualizing data distributions: `stroke_distribution`,
 `age_distribution_by_gender`,
  `numeric_variables_by_stroke`, `dynamic_plots`
- Correlation analysis: `point_biserial_heatmap`
- Q-Q plots: `qq_plots`
- Model evaluation: `cv_results`, `multi_conf_matrix`

The module also defines a custom color palette
and background color for consistent
styling across visualizations.
"""

import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D
from scipy.stats import pointbiserialr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CITRINE = "#e9cb0c"
NAPLES = "#ffd470"
CREAM = "#f3f6cb"
APPLE = "#9ea300"
MOSS = "#555610"
BACKGROUND_COLOR = "white"
ml_colors = [MOSS, APPLE, CREAM, NAPLES, CITRINE]
cmap = plt.cm.colors.ListedColormap(ml_colors)


def stroke_distribution(df: pd.DataFrame) -> None:
    """
    Generates a bar plot showing the distribution of
    stroke samples in a dataframe.

    Args:
      df: pandas DataFrame containing the data with a 'stroke' column.

    Returns:
      None. Displays the plot.
    """
    fig = plt.figure(figsize=(12, 12), dpi=150, facecolor=BACKGROUND_COLOR)
    gs = fig.add_gridspec(4, 3)
    gs.update(wspace=0.1, hspace=0.4)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(BACKGROUND_COLOR)
    ax0.tick_params(axis='y', left=False)
    ax0.get_yaxis().set_visible(False)
    for s in ["top", "right", "left"]:
        ax0.spines[s].set_visible(False)

    heights, bins = np.histogram(df['stroke'], bins=2)
    bin_width = np.diff(bins)[0]
    bin_pos = (bins[:-1] + bin_width / 2)

    total_samples = len(df)
    stroke_counts = df['stroke'].value_counts()
    percentages = (stroke_counts / total_samples * 100).round(2)

    ax0.bar(bin_pos, heights, width=bin_width, edgecolor=MOSS,
            linewidth=2, color=[APPLE, CITRINE])

    for i, p in enumerate(percentages):
        ax0.text(bin_pos[i], heights[i], f'{p}%', ha='center', va='bottom')

    ax0.grid(which='major', axis='x', zorder=0, color=APPLE,
             linestyle=':', dashes=(1, 5))
    ax0.set_xlabel('Stroke')
    ax0.set_title('Distribution of Stroke Samples', fontsize=17,
                  fontweight='light', fontfamily='monospace')

    legend_elements = [Line2D([0], [0], color=CITRINE, lw=1, label='Stroke'),
                       Line2D([0], [0], color=APPLE, lw=1, label='No Stroke')]
    ax0.legend(handles=legend_elements)
    ax0.set_xticks([0, 1])
    plt.show()


def age_distribution_by_gender(df: pd.DataFrame) -> None:
    """
    Generates a bar plot showing the age distribution by gender in a dataframe.

    Args:
      df: pandas DataFrame containing the data with 'age' and 'gender' columns.

    Returns:
      None. Displays the plot.
    """
    fig = plt.figure(figsize=(12, 18), dpi=150, facecolor=BACKGROUND_COLOR)
    gs = fig.add_gridspec(4, 3)
    gs.update(wspace=0.1, hspace=0.4)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(BACKGROUND_COLOR)
    ax0.tick_params(axis='y', left=False)
    ax0.get_yaxis().set_visible(False)
    for s in ["top", "right", "left"]:
        ax0.spines[s].set_visible(False)

    heights, bins = np.histogram(df.query('gender=="Male"').age,
                                 density=True, bins=20)
    bin_width = np.diff(bins)[0]
    bin_pos = (bins[:-1] + bin_width / 2)

    ax0.bar(bin_pos, heights, width=bin_width, edgecolor=MOSS,
            color=CITRINE, linewidth=2)

    heights, bins = np.histogram(df.query('gender=="Female"').age,
                                 density=True, bins=20)
    heights *= -1

    ax0.bar(bin_pos, heights, width=bin_width, edgecolor=MOSS,
            color=APPLE, linewidth=2)

    total_males = len(df.query('gender=="Male"'))
    total_females = len(df.query('gender=="Female"'))
    male_percentage = (total_males / len(df) * 100)
    female_percentage = (total_females / len(df) * 100)

    ax0.text(0.5, 0.94, f'{male_percentage:.2f}%', ha='center',
             va='center', transform=ax0.transAxes)
    ax0.text(0.5, 0.06, f'{female_percentage:.2f}%', ha='center',
             va='center', transform=ax0.transAxes)

    ax0.grid(which='major', axis='x', zorder=0, color=APPLE,
             linestyle=':', dashes=(1, 5))
    ax0.set_xlabel('Age')
    ax0.text(0, 0.022, 'Age Distribution by Gender',
             fontsize=17, fontweight='light', fontfamily='monospace')

    legend_elements = [Line2D([0], [0], color=CITRINE, lw=1, label='Male'),
                       Line2D([0], [0], color=APPLE, lw=1, label='Female')]

    ax0.legend(handles=legend_elements)
    plt.show()


def numeric_variables_by_stroke(df: pd.DataFrame) -> None:
    """
    Generates a figure with three subplots showing the distribution of
    numeric variables ('age', 'avg_glucose_level', 'bmi')
    by stroke status.

    Args:
      df: pandas DataFrame containing the data with 'age',
      'avg_glucose_level', 'bmi', and 'stroke' columns.

    Returns:
      None. Displays the plot.
    """
    fig = plt.figure(figsize=(12, 12), dpi=150,
                     facecolor=BACKGROUND_COLOR)
    gs = fig.add_gridspec(4, 3)
    gs.update(wspace=0.1, hspace=0.4)

    ax = [None for _ in range(3)]

    for plot in range(3):
        ax[plot] = fig.add_subplot(gs[0, plot])
        ax[plot].set_facecolor(BACKGROUND_COLOR)
        ax[plot].tick_params(axis='y', left=False)
        ax[plot].get_yaxis().set_visible(False)
        for s in ["top", "right", "left"]:
            ax[plot].spines[s].set_visible(False)

    s = df[df['stroke'] == 1]
    ns = df[df['stroke'] == 0]

    for plot, feature in enumerate(['age', 'avg_glucose_level', 'bmi']):
        sns.kdeplot(s[feature], ax=ax[plot],
                    color=CITRINE, fill=True, linewidth=2, ec=MOSS,
                    alpha=0.9, zorder=3, legend=False)
        sns.kdeplot(ns[feature], ax=ax[plot],
                    color=APPLE, fill=True, linewidth=2, ec=MOSS,
                    alpha=0.9, zorder=3, legend=False)
        ax[plot].grid(which='major', axis='x',
                      zorder=0, color=APPLE,
                      linestyle=':', dashes=(1, 5))

    ax[0].text(-20, 0.056, 'Numeric Variables by Stroke & No Stroke',
               fontsize=17, fontweight='light', fontfamily='monospace')
    ax[0].text(-20, 0.05, 'Age looks to be a prominent factor',
               fontsize=13, fontweight='light', fontfamily='monospace')

    legend_elements = [Line2D([0], [0], color=CITRINE,
                              lw=1, label='Stroke'),
                       Line2D([0], [0], color=APPLE,
                              lw=1, label='No Stroke')]

    ax[0].legend(handles=legend_elements)
    plt.show()


def point_biserial_heatmap(df: pd.DataFrame,
                           target_col: str, cmap: str = cmap) -> plt.Axes:
    """
    Generates a heatmap of point biserial correlations between a
    target boolean column
    and all other numeric columns in a DataFrame,
    ranked by absolute correlation size.

    Args:
      df: pandas DataFrame containing the data.
      target_col: Name of the boolean target column in the DataFrame.
      cmap: The colormap for the heatmap.

    Returns:
      A matplotlib Axes object containing the heatmap.
    """

    y = df[target_col]
    numeric_df = df.select_dtypes(include=['number'])

    if target_col in numeric_df:
        numeric_df = numeric_df.drop(target_col, axis=1)

    correlations = numeric_df.apply(lambda x: pointbiserialr(x, y)[0])
    correlations = correlations.abs().sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations.to_frame(), annot=True,
                cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'Point Biserial Correlation with {target_col} (Ranked)')
    plt.show()


def dynamic_plots(data: pd.DataFrame,
                  features: Optional[List[str]] = None,
                  max_cols: int = 3,
                  figsize: Tuple[int, int] = (15, 12),
                  plot_type: str = 'histplot') -> None:
    """
    Creates subplots dynamically based on features,
    with a maximum columns per row.
    Allows choosing between histograms and boxplots.

    Args:
        data (pd.DataFrame): The DataFrame containing your data.
        features (list, optional): List of features to plot.
        If None, uses all columns.
        max_cols (int, optional): Maximum columns per row. Defaults to 3.
        figsize (tuple, optional): Figure size (width, height).
        Defaults to (15, 12).
        plot_type (str, optional): Type of plot to create.
                                   'histplot' for histograms (default),
                                    'boxplot' for boxplots.
    """

    if features is None:
        features = data.columns.tolist()

    num_features = len(features)
    num_rows = math.ceil(num_features / max_cols)

    fig, axes = plt.subplots(num_rows, max_cols, figsize=figsize)

    if num_rows > 1:
        axes = axes.flatten()

    for index, feature in enumerate(features):
        if num_rows == 1:
            ax = axes[index]
        else:
            ax = axes[index]

        if plot_type == 'histplot':
            sns.histplot(data[feature], color=CITRINE, ax=ax,
                         kde=True, edgecolor=CREAM, linewidth=2)
            kde_line = ax.lines[0]
            kde_line.set_color(MOSS)

        elif plot_type == 'boxplot':
            sns.boxplot(data[feature], color=APPLE, ax=ax,
                        whiskerprops=dict(color=MOSS),
                        capprops=dict(color=MOSS),
                        flierprops=dict(markeredgecolor=APPLE))
        else:
            raise ValueError('Invalid plot_type. Choose \'histplot\''
                             ' or \'boxplot\'.')

    if num_features < num_rows * max_cols:
        for i in range(num_features, num_rows * max_cols):
            axes[i].set_axis_off()

    plt.tight_layout()
    plt.show()


def qq_plots(data: pd.DataFrame | np.ndarray,
             variable_names: list[str]) -> None:
    """
    Generates multiple Q-Q plots for a list of variables in a dataset.

    Args:
      data: pandas DataFrame or numpy array containing the data.
      variable_names: List of variable names to plot.

    Returns:
      None. Displays the Q-Q plots.
    """

    num_plots = len(variable_names)
    num_cols = min(num_plots, 3)
    num_rows = int(np.ceil(num_plots / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, variable_name in enumerate(variable_names):
        if isinstance(data, pd.DataFrame):
            variable_data = data[variable_name].values
        else:
            variable_data = data[:, i]

        sm.qqplot(variable_data, line='45', fit=True, ax=axes[i])
        axes[i].set_title(f'Q-Q Plot for {variable_name}')

        axes[i].get_lines()[0].set_markerfacecolor(CITRINE)
        axes[i].get_lines()[0].set_markeredgecolor(CITRINE)
        for child in axes[i].get_children():
            if isinstance(child, plt.matplotlib.lines.Line2D):
                child.set_color(MOSS)

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def cv_results(results_df: pd.DataFrame) -> None:
    """
    Plots the distribution of cross-validation scores as a histogram.

    Args:
      results_df: DataFrame containing the cross-validation scores.
      APPLE: Color for the first set of bars.
      CITRINE: Color for the second set of bars.
    """
    bins = np.linspace(start=0.35, stop=1.0, num=50)

    results_df.plot.hist(bins=bins, edgecolor=MOSS, color=[APPLE, CITRINE])

    plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
    plt.xlabel("Accuracy (%)")
    plt.title("Distribution of the CV scores")
    plt.show()


def multi_conf_matrix(models: List,
                      X: pd.DataFrame,
                      y: pd.Series,
                      display_labels: List[str]):
    """
    Plots multiple confusion matrices for a given list of models.

    Args:
      models: A list of trained machine learning models.
      X: The feature matrix as a Pandas DataFrame.
      y: The target variable array as a Pandas Series.
      display_labels: A list of string labels for the classes in
                      the confusion matrix.
    """
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))

    for i, model in enumerate(models):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=display_labels)
        disp.plot(ax=axes[i], cmap=cmap)
        axes[i].set_title(model.__class__.__name__, fontsize=14,
                          fontweight='bold')
        for text in disp.text_.ravel():
            text.set_fontsize(18)
            text.set_fontweight('bold')

    plt.tight_layout()
    plt.show()
