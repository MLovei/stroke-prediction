"""
This module provides statistical utility functions for data analysis.
"""

from typing import List, Tuple
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, shapiro, t


def check_normality(df: pd.DataFrame, cols: List[str]) -> None:
    """
    Checks the normality of specified numerical distributions
    in a pandas DataFrame using the Shapiro-Wilkes test.

    Args:
      df: pandas DataFrame containing the data.
      cols: List of column names to check for normality.

    Returns:
      None. Prints the Shapiro-Wilkes test results and
      a comment about the p-value for each specified column.
    """

    for col in cols:
        if col in df.columns:
            statistic, p_value = shapiro(df[col].dropna())
            print(f"Shapiro-Wilkes test for {col}:")
            print(f"  - Statistic: {statistic:.3f}")
            print(f"  - P-value: {p_value:.3f}")

            alpha = 0.05
            if p_value > alpha:
                print(
                    f"  - Comment: The data for '{col}' appears to"
                    f" be normally distributed (fail to reject H0)."
                )
            else:
                print(
                    f"  - Comment: The data for '{col}' likely does"
                    f" not follow a normal distribution (reject H0)."
                )

            print("\n")
        else:
            print(f"Column '{col}' not found in the DataFrame.")


def chi_squared_test(observed: List[int],
                     expected: List[int]) -> Tuple[float, float, int]:
    """
    Performs a Chi-squared statistical test.

    Args:
        observed: A list of observed frequencies.
        expected: A list of expected frequencies.

    Returns:
        A tuple containing the Chi-squared statistic, the p-value,
        and the degrees of freedom.
    """

    chi2, p, dof, _ = chi2_contingency([observed, expected])

    print("Chi-squared Test Results:")
    print(f"  Chi-squared statistic: {chi2:.2f}")
    print(f"  P-value: {p:.3f}")
    print(f"  Degrees of freedom: {dof}")

    # Interpretation
    if p < 0.05:
        print(
            "\nInterpretation: There is a statistically significant"
            " difference between the observed and expected frequencies.")
    else:
        print(
            "\nInterpretation: There is no statistically significant"
            " difference between the observed and expected frequencies.")

    return chi2, p, dof


def test_median_difference(df: pd.DataFrame,
                           variable: str,
                           group_variable: str) -> None:
    """
    Performs the Mann-Whitney U test to compare the median values of a
    variable between two groups.

    Args:
      df: pandas DataFrame containing the data.
      variable: Name of the variable to compare (must be numeric).
      group_variable: Name of the grouping variable (must be categorical).

    Returns:
      None. Prints the Mann-Whitney U test results and a comment about
      the median difference.
    """

    if variable not in df.columns or group_variable not in df.columns:
        print("Variable or group variable not found in the DataFrame.")
        return

    group1 = df[df[group_variable] ==
                df[group_variable].unique()[0]][variable].dropna()
    group2 = df[df[group_variable] ==
                df[group_variable].unique()[1]][variable].dropna()

    statistic, p_value = mannwhitneyu(group1, group2)

    print(f"Mann-Whitney U test for '{variable}'"
          f" between groups in '{group_variable}':")
    print(f"  - Statistic: {statistic:.3f}")
    print(f"  - P-value: {p_value:.3f}")

    alpha = 0.05
    if p_value > alpha:
        print(f"  - Comment: No significant difference in median"
              f" '{variable}' between groups (fail to reject H0).")
    else:
        print(f"  - Comment: Significant difference in median"
              f" '{variable}' between groups (reject H0).")


def calculate_confidence_intervals(df: pd.DataFrame,
                                   variable: str, group_variable: str = None,
                                   confidence: float = 0.95) -> None:
    """
    Calculates and prints confidence intervals for a given variable,
    optionally grouped by another variable.

    Args:
      df: pandas DataFrame containing the data.
      variable: Name of the variable to calculate
      confidence intervals for (must be numeric).
      group_variable: Optional name of the
      grouping variable (must be categorical).
      confidence: Confidence level for the interval (default is 0.95).
    """

    if variable not in df.columns:
        print(f"Variable '{variable}' not found in the DataFrame.")
        return

    if group_variable and group_variable not in df.columns:
        print(f"Group variable '{group_variable}' not found in the DataFrame.")
        return

    def ci_for_group(data: pd.Series) -> tuple:
        """Helper function to calculate CI for a single group."""
        mean = data.mean()
        std_err = data.sem()
        margin_of_error = std_err * t.ppf((1 + confidence) / 2, len(data) - 1)
        return mean, mean - margin_of_error, mean + margin_of_error

    if group_variable:
        groups = df[group_variable].unique()
        for group in groups:
            group_data = df[df[group_variable] == group][variable].dropna()
            mean, lower_ci, upper_ci = ci_for_group(group_data)
            print(f"Confidence Intervals for {variable} "
                  f"({group_variable} = {group}):")
            print(f"- Mean: {mean:.2f}")
            print(f"- {confidence*100:.0f}% CI:"
                  f" ({lower_ci:.2f}, {upper_ci:.2f})")
            print("\n")
    else:
        mean, lower_ci, upper_ci = ci_for_group(df[variable].dropna())
        print(f"Confidence Intervals for {variable} (Entire Population):")
        print(f"- Mean: {mean:.2f}")
        print(f"- {confidence*100:.0f}% CI: ({lower_ci:.2f}, {upper_ci:.2f})")
        print("\n")
