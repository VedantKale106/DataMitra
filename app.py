from flask import Flask, request, render_template, send_file
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import pickle  # Import for serialization

app = Flask(__name__)

# Initialize global variables
df = pd.DataFrame()
history = []
encoders = {}  # Dictionary to store encoders for each column
scalers = {}   # Dictionary to store scalers for each column

# Helper functions for preprocessing
def encode_categorical_columns(df, column):
    if column in df.columns and df[column].dtype in ['object', 'category']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le  # Store the encoder
    return df

def scale_numeric_column(df, column):
    if column in df.columns and df[column].dtype in ['float64', 'int64']:
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[[column]])
        scalers[column] = scaler  # Store the scaler
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
        history.pop()  # Remove the last operation
        df = history[-1]  # Revert to the previous state

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
    if plot_type == 'histogram':
        df[x_column].hist(ax=ax)
    elif plot_type == 'scatter' and y_column:
        df.plot.scatter(x=x_column, y=y_column, ax=ax)
    elif plot_type == 'box':
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
            # Numerical statistics
            stats['Max'] = df[column].max()
            stats['Min'] = df[column].min()
            stats['Mean'] = df[column].mean()
            stats['Standard Deviation'] = df[column].std()
            stats['Variance'] = df[column].var()
            stats['Median'] = df[column].median()
            stats['Mode'] = df[column].mode()[0]
        else:
            # Categorical statistics
            stats['Mode'] = df[column].mode()[0]
            stats['Top 5 Frequent Values'] = df[column].value_counts().head().to_dict()

    return stats


def export_encoders_and_scalers():
    """Export encoders and scalers as a pickle file."""
    # Create a dictionary with encoders and scalers
    data = {'encoders': encoders, 'scalers': scalers}
    # Serialize using pickle
    pickle_data = pickle.dumps(data)
    # Use BytesIO to handle the binary stream
    file_object = BytesIO(pickle_data)
    # Return the file as an attachment
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
    column = ''  # Initialize column variable
    plot_image = None
    dataset_details = None

    if request.method == 'POST':
        if 'upload_file' in request.files:
            file = request.files['upload_file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                history = [df.copy()]  # Initialize history stack

        if 'handle_missing_values' in request.form:
            column = request.form['column']
            history.append(df.copy())  # Save the current state
            df = handle_missing_values(df, column)

        if 'encode_categorical_columns' in request.form:
            column = request.form['column']
            history.append(df.copy())  # Save the current state
            df = encode_categorical_columns(df, column)

        if 'drop_duplicates' in request.form:
            history.append(df.copy())  # Save the current state
            df = drop_duplicates(df)

        if 'scale_numeric_column' in request.form:
            column = request.form['column']
            history.append(df.copy())  # Save the current state
            df = scale_numeric_column(df, column)

        if 'delete_column' in request.form:
            column = request.form['column']
            history.append(df.copy())  # Save the current state
            df = delete_column(df, column)

        if 'undo' in request.form:
            undo_last_operation()

        if 'export_csv' in request.form:
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name='processed_dataset.csv', mimetype='text/csv')

        if 'export_encoders_scalers' in request.form:
            return export_encoders_and_scalers()

        if 'show_column_statistics' in request.form:
            column = request.form['column']
            column_stats = get_column_statistics(df, column) if column and not df.empty else {}

        if 'show_dataset_details' in request.form:
            dataset_details = calculate_dataset_details(df) if not df.empty else {}

        if 'plot_type' in request.form:
            plot_type = request.form['plot_type']
            x_column = request.form['x_column'] if 'x_column' in request.form else None
            y_column = request.form['y_column'] if 'y_column' in request.form else None
            if x_column in df.columns and (plot_type != 'scatter' or (x_column in df.columns and y_column in df.columns)):
                plot_image = generate_plot(df, plot_type, x_column, y_column)

    columns = df.columns.tolist() if not df.empty else []
    sample_size = min(10, len(df)) if not df.empty else 0
    preview_df = df.sample(n=sample_size) if sample_size > 0 else pd.DataFrame()

    return render_template('index.html',
                           preview=preview_df.to_html(classes='table table-striped', index=False),
                           columns=columns,
                           column_stats=column_stats,
                           column_name=column,
                           plot_image=plot_image,
                           dataset_details=dataset_details)

if __name__ == '__main__':
    app.run(debug=True)
