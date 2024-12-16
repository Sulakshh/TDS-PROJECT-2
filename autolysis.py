import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
import os

def validate_environment():
    if "AIPROXY_TOKEN" not in os.environ:
        print("Error: AIPROXY_TOKEN environment variable is missing.")
        sys.exit(1)

def create_output_structure(base_name):
    # Create base directory
    if not os.path.exists(base_name):
        os.makedirs(base_name)

    return base_name

def analyze_dataset(file_path):
    # Extract dataset name
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = create_output_structure(dataset_name)

    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        sys.exit(1)

    # Describe the dataset
    describe_data(df, output_dir)

    # Visualize the dataset
    create_visualizations(df, output_dir)

    # Predict a target if possible
    if detect_target_column(df):
        predict_target(df, output_dir)
    
    # Generate the README.md file
    generate_readme(output_dir)

# Step 1: Descriptive Analysis

def describe_data(df, output_dir):
    print("\nBasic Information About the Dataset:\n")
    info_buf = []
    df.info(buf=info_buf)
    info_str = "\n".join(info_buf)
    print(info_str)

    description = df.describe(include='all').transpose()
    description['missing_values'] = df.isnull().sum()
    description['missing_percentage'] = description['missing_values'] / len(df) * 100

    print("\nDescriptive Statistics:\n")
    print(tabulate(description, headers="keys", tablefmt="fancy_grid"))

    global readme_content
    readme_content += (
        "## Dataset Overview\n\n"
        "This dataset was analyzed to uncover meaningful patterns and insights. Below is an overview of its "
        "descriptive statistics, highlighting missing values, distributions, and key metrics for numeric columns:\n\n"
        + description.to_markdown() + "\n\n"
    )
    readme_content += (
        "From this summary, we observe potential areas for data cleaning and preparation, especially where missing "
        "values are significant.\n\n"
    )

# Step 2: Visualization

def create_visualizations(df, output_dir):
    global readme_content

    try:
        # Generate a correlation heatmap
        numeric_cols = df.select_dtypes(include=['number'])
        if not numeric_cols.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            heatmap_path = os.path.join(output_dir, "heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            readme_content += (
                "### Correlation Heatmap\n\n"
                "The following heatmap visualizes the relationships between numerical variables in the dataset:\n\n"
                f"![Correlation Heatmap]({heatmap_path})\n\n"
            )

        # Generate distribution plots
        for col in numeric_cols.columns[:3]:  # Limit to 3 plots
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            dist_path = os.path.join(output_dir, f"{col}_distribution.png")
            plt.savefig(dist_path)
            plt.close()
            readme_content += (
                f"### {col} Distribution\n\n"
                f"The following plot shows the distribution of `{col}`, revealing patterns such as skewness or outliers:\n\n"
                f"![{col} Distribution]({dist_path})\n\n"
            )
    except Exception as e:
        print(f"Error creating visualizations: {e}")

# Step 3: Detect and Predict Target Column

def detect_target_column(df):
    numeric_cols = df.select_dtypes(include=['number'])
    return numeric_cols.shape[1] > 1  # Ensure there are enough numeric columns for prediction

def predict_target(df, output_dir):
    global readme_content

    numeric_cols = df.select_dtypes(include=['number'])
    target_col = numeric_cols.columns[-1]  # Assume last numeric column as target

    features = numeric_cols.iloc[:, :-1]  # All but the last numeric column
    target = numeric_cols[target_col]

    # Drop rows with missing values
    data = pd.concat([features, target], axis=1).dropna()

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    readme_content += (
        "## Predictive Model Results\n\n"
        "A regression model was trained to predict `{target_col}` based on other numerical features. Below are the results:\n\n"
        f"- **Mean Squared Error (MSE):** {mse:.2f}\n"
        f"- **R-squared (R²):** {r2:.2f}\n\n"
        "These results highlight the model's effectiveness in capturing relationships within the dataset.\n\n"
    )

# Step 4: Generate README.md

def generate_readme(output_dir):
    global readme_content
    readme_content = (
        "# Automated Dataset Analysis\n\n"
        "This analysis aims to uncover insights and patterns from the dataset, leveraging descriptive statistics, "
        "visualizations, and predictive modeling. Below is a comprehensive breakdown:\n\n"
    ) + readme_content + (
        "### Recommendations\n\n"
        "- Consider addressing missing values in key columns to improve data quality.\n"
        "- Utilize insights from the visualizations to refine future data collection efforts.\n"
        "- Apply the predictive model for operational decision-making in relevant domains.\n"
    )

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"README.md file generated successfully at {readme_path}.")

# Main Function
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    validate_environment()

    dataset_path = sys.argv[1]

    if not os.path.isfile(dataset_path):
        print(f"Error: File {dataset_path} does not exist.")
        sys.exit(1)

    readme_content = """# Automated Dataset Analysis\n\n"""
    analyze_dataset(dataset_path)
