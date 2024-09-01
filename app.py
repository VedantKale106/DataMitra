from flask import Flask, request, render_template, send_file
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import pickle
from scipy.stats import zscore

app = Flask(__name__)

# Initialize global variables
df = pd.DataFrame()
history = []
encoders = {}
scalers = {}

# Helper functions for preprocessing

def remove_outliers(df, column, method='zscore'):
    if column not in df.columns:
        return df

    if method == 'zscore':
        df['zscore'] = zscore(df[column])
        df = df[(df['zscore'] >= -3) & (df['zscore'] <= 3)]
        df.drop(columns=['zscore'], inplace=True)
    elif method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df

def encode_categorical_columns(df, column):
    if column in df.columns and df[column].dtype in ['object', 'category']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le
    return df

def scale_numeric_column(df, column):
    if column in df.columns and df[column].dtype in ['float64', 'int64']:
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[[column]])
        scalers[column] = scaler
    return df

def handle_missing_values(df, column):
    if column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def drop_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def delete_column(df, column):
    if column in df.columns:
        df.drop(column, axis=1, inplace=True)
    return df

def undo_last_operation():
    global df, history
    if len(history) > 1:
        history.pop()
        df = history[-1]

def calculate_dataset_details(df):
    num_numerical = df.select_dtypes(include=['float64', 'int64']).shape[1]
    num_categorical = df.select_dtypes(include=['object', 'category']).shape[1]
    return {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'num_numerical': num_numerical,
        'num_categorical': num_categorical,
        'column_names': df.columns.tolist()
    }

def generate_plot(df, plot_type, x_column, y_column=None):
    fig, ax = plt.subplots()
    if plot_type == 'histogram' and x_column in df.columns:
        df[x_column].hist(ax=ax)
    elif plot_type == 'scatter' and x_column in df.columns and y_column in df.columns:
        df.plot.scatter(x=x_column, y=y_column, ax=ax)
    elif plot_type == 'box' and x_column in df.columns:
        df[[x_column]].boxplot(ax=ax)

    output = BytesIO()
    plt.savefig(output, format='png')
    plt.close(fig)
    output.seek(0)
    plot_data = output.getvalue()
    encoded_plot = base64.b64encode(plot_data).decode('utf-8')
    return encoded_plot

def get_column_statistics(df, column):
    stats = {}
    if column in df.columns:
        stats['Unique Values Count'] = df[column].nunique()
        stats['Null Values Count'] = df[column].isnull().sum()
        stats['Unique Values'] = df[column].unique().tolist()

        if df[column].dtype in ['float64', 'int64']:
            stats['Max'] = df[column].max()
            stats['Min'] = df[column].min()
            stats['Mean'] = df[column].mean()
            stats['Standard Deviation'] = df[column].std()
            stats['Variance'] = df[column].var()
            stats['Median'] = df[column].median()
            stats['Mode'] = df[column].mode()[0]
        else:
            stats['Mode'] = df[column].mode()[0]
            stats['Top 5 Frequent Values'] = df[column].value_counts().head().to_dict()

    return stats

def delete_selected_value(df, column, value):
    if column in df.columns:
        df = df[df[column] != value]
    return df

def replace_values_in_column(df, column, value_to_replace, new_value):
    if column in df.columns:
        df[column] = df[column].replace(value_to_replace, new_value)
    return df

def validate_dataframe(df):
    """Perform comprehensive validation on the DataFrame."""
    
    errors = []
    
    # Check for missing values
    if df.isnull().values.any():
        missing_info = df.isnull().sum()
        missing_columns = missing_info[missing_info > 0].index.tolist()
        errors.append(f"Missing values found in columns: {', '.join(missing_columns)}")

    # Check for duplicate rows
    if df.duplicated().any():
        errors.append("Duplicate rows found in the DataFrame.")

    # Check for unique values in categorical columns (optional)
    for column in df.select_dtypes(include=['object', 'category']).columns:
        unique_values_count = df[column].nunique()
        if unique_values_count > 50:  # Arbitrary threshold for example
            errors.append(f"Column '{column}' has too many unique values ({unique_values_count}).")

    if errors:
        raise ValueError("DataFrame validation failed with the following issues:\n" + "\n".join(errors))
    print("DataFrame validation passed.")

def export_encoders_and_scalers():
    data = {'encoders': encoders, 'scalers': scalers}
    pickle_data = pickle.dumps(data)
    file_object = BytesIO(pickle_data)
    file_object.seek(0)  # Ensure the pointer is at the start of the BytesIO stream
    return send_file(file_object, download_name='encoders_scalers.pkl', as_attachment=True, mimetype='application/octet-stream')

@app.route('/user_manual')
def user_manual():
    return render_template('user_manual.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    global df, history, encoders, scalers
    column_stats = {}
    column = ''
    plot_image = None
    dataset_details = None
    unique_values = None
    message = ''

    if request.method == 'POST':
        if 'upload_file' in request.files:
            file = request.files['upload_file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                history = [df.copy()]

        if 'handle_missing_values' in request.form:
            column = request.form.get('column')
            if column in df.columns:
                history.append(df.copy())
                df = handle_missing_values(df, column)
            else:
                message = f"Column '{column}' does not exist in the DataFrame."

        if 'encode_categorical_columns' in request.form:
            column = request.form.get('column')
            if column in df.columns:
                history.append(df.copy())
                df = encode_categorical_columns(df, column)
            else:
                message = f"Column '{column}' does not exist in the DataFrame."

        if 'drop_duplicates' in request.form:
            history.append(df.copy())
            df = drop_duplicates(df)

        if 'scale_numeric_column' in request.form:
            column = request.form.get('column')
            if column in df.columns:
                history.append(df.copy())
                df = scale_numeric_column(df, column)
            else:
                message = f"Column '{column}' does not exist in the DataFrame."

        if 'delete_column' in request.form:
            column = request.form.get('column')
            if column in df.columns:
                history.append(df.copy())
                df = delete_column(df, column)
            else:
                message = f"Column '{column}' does not exist in the DataFrame."

        if 'undo' in request.form:
            undo_last_operation()

        if 'export_csv' in request.form:
            if not df.empty:
                try:
                    buffer = BytesIO()
                    df.to_csv(buffer, index=False)
                    buffer.seek(0)
                    return send_file(buffer, as_attachment=True, download_name='processed_dataset.csv', mimetype='text/csv')
                except Exception as e:
                    print(f"Error exporting CSV: {e}")  # Debugging output
                    message = "An error occurred while exporting the CSV."
            else:
                message = "No data available to export."



        if 'export_encoders_scalers' in request.form:
            return export_encoders_and_scalers()

        if 'show_column_statistics' in request.form:
            column = request.form.get('column')
            if column in df.columns:
                column_stats = get_column_statistics(df, column)

        if 'show_dataset_details' in request.form:
            dataset_details = calculate_dataset_details(df) if not df.empty else {}

        if 'plot_type' in request.form:
            plot_type = request.form.get('plot_type')
            x_column = request.form.get('x_column')
            y_column = request.form.get('y_column')
            if x_column in df.columns and (plot_type != 'scatter' or y_column in df.columns):
                plot_image = generate_plot(df, plot_type, x_column, y_column)

        if 'column' in request.form:
            column = request.form.get('column')
            if column in df.columns:
                unique_values = df[column].unique().tolist()

        if 'delete_selected_value' in request.form:
            column = request.form.get('column')
            unique_value = request.form.get('unique_value')
            if column in df.columns and unique_value:
                history.append(df.copy())
                df = delete_selected_value(df, column, unique_value)
                unique_values = df[column].unique().tolist()

        if 'remove_outliers' in request.form:
            column = request.form.get('outlier_column')
            method = request.form.get('outlier_method')
            if column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    history.append(df.copy())
                    df = remove_outliers(df, column, method)
                else:
                    message = f"Column '{column}' is not numeric. Outlier removal can only be applied to numeric columns."

        if 'replace_values' in request.form:
            column = request.form.get('replace_column')
            value_to_replace = request.form.get('value_to_replace')
            new_value = request.form.get('new_value')
            if column in df.columns:
                history.append(df.copy())
                df = replace_values_in_column(df, column, value_to_replace, new_value)

        if 'check_validation' in request.form:
            try:
                validate_dataframe(df)  # Validate the DataFrame
                message = "DataFrame validation passed."
            except ValueError as e:
                message = str(e)

    columns = df.columns.tolist() if not df.empty else []
    sample_size = min(10, len(df)) if not df.empty else 0
    preview_df = df.sample(n=sample_size) if sample_size > 0 else pd.DataFrame()

    return render_template('index.html',
                           preview=preview_df.to_html(classes='table table-striped', index=False),
                           columns=columns,
                           column_stats=column_stats,
                           unique_values=unique_values,
                           column_name=column,
                           plot_image=plot_image,
                           dataset_details=dataset_details,
                           message=message)

if __name__ == '__main__':
    app.run(debug=True)
