# DataMitra

This Flask application allows users to upload a CSV file and perform various data preprocessing operations, such as handling missing values, encoding categorical columns, scaling numerical columns, and more. The app also provides options to view dataset details, column statistics, generate plots, and export encoders, scalers, and processed datasets.

## Features

- **Upload CSV**: Allows users to upload a CSV file to be processed.
- **Handle Missing Values**: Fills missing values with mean for numerical columns and mode for categorical columns.
- **Encode Categorical Columns**: Converts categorical columns to numerical values using Label Encoding.
- **Scale Numeric Columns**: Scales numeric columns using StandardScaler.
- **Drop Duplicates**: Removes duplicate rows from the dataset.
- **Delete Columns**: Allows users to delete specified columns.
- **Remove Selected Unique Value**: Provides an option to remove specific unique values from a selected column.
- **Replace Values in a Column**: Enables users to replace specified values in a column with new values.
- **Remove Outliers**: Provides options to detect and remove outliers from a selected column using Z-Score or IQR (Interquartile Range) methods.
- **Undo Operation**: Reverts to the previous state of the dataset.
- **Export CSV**: Downloads the processed dataset as a CSV file.
- **Export Encoders and Scalers**: Downloads encoders and scalers as a pickle file.
- **View Column Statistics**: Displays various statistics for a selected column.
- **Show Dataset Details**: Shows details about the dataset, including the number of rows, columns, and types of data.
- **Generate Plots**: Creates histograms, scatter plots, and box plots for data visualization.
- **Check Validation**: Validates the data within the dataset for consistency and correctness.

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

### User Manual

For a detailed guide on how to use the application, click on the 'User Manual' link in the app or navigate to `http://127.0.0.1:5000/user_manual`.

### Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code follows the project's coding standards and includes relevant tests.

### Developers

This app is developed by Vedant Kale and Niraj Suryavanshi

### Acknowledgments

- Flask for the web framework.
- Pandas and Scikit-Learn for data manipulation and preprocessing.
- Matplotlib and Seaborn for data visualization.

### Contact

For any questions or feedback, please reach out to [your-email@example.com](mailto:vedant.kale22@pccoepune.org).
