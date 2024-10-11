import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import io
import base64
import flask
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize Flask server and Dash app
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# Ensure the server variable is directly referenced for deployment
application = app.server

# Function to generate dataset based on user inputs
def generate_data(n_samples=100, noise=50):
    X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=42)
    return X, y

# Initial dataset (global fallback dataset)
initial_X, initial_y = generate_data()

# Function to plot the raw dataset and the regression line
def plot_regression_line(X, y, model=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(X, y, s=20, c='black', label="Data Points")

    # Plot the regression line if a model is provided
    if model is not None:
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_range)
        ax.plot(X_range, y_pred, color='red', label="Regression Line")
        
    ax.set_xlabel("X", fontsize=26)
    ax.set_ylabel("Y", fontsize=26)
    
    # Customize specific spines
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=20, labelcolor='black', width=4, length=10)

    ax.legend(fontsize=20).get_frame().set_linewidth(0)  # Remove the legend border

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    # Encode the plot as a base64 string
    return base64.b64encode(buf.read()).decode('utf-8')

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("Model Evaluation in Simple Linear Regression", style={'textAlign': 'center'}),
    html.Label("Number of Samples:"),
    dcc.Input(
        id='num-samples-input',
        type='number',
        value=100,  # Default number of samples
        min=10,
        step=1,
        style={'width': '20%'}
    ),
    html.Br(),
    html.Label("Noise Level:"),
    dcc.Input(
        id='noise-input',
        type='number',
        value=50,  # Default noise level
        min=0,
        step=1,
        style={'width': '20%'}
    ),
    html.Br(),
    html.Label("Select Model Validation Technique:"),
    dcc.Dropdown(
        id='validation-dropdown',
        options=[
            {'label': 'Train/Test on Same Dataset', 'value': 'same_dataset'},
            {'label': 'Train/Test Split', 'value': 'train_test_split'},
            {'label': 'K-Fold Cross-Validation', 'value': 'k_fold'}
        ],
        value=None,  # No default value selected
        style={'width': '50%'}
    ),
    html.Br(),
    html.Label("Enter the value of K (for K-Fold Cross-Validation):"),
    dcc.Input(
        id='k-value-input',
        type='number',
        value=5,  # Default value for K
        min=2,
        step=1,
        style={'width': '20%'},
        disabled=True  # Initially disabled
    ),
    html.Br(),
    html.Br(),
    html.Div(id='output-container', style={'marginTop': 20}),
    html.Hr(),
    html.H2("Regression Line and Dataset Plot", style={'textAlign': 'center'}),
    html.Img(
        id='regression-plot',
        style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '60%'}
    ),
    html.H2("MSE vs K-Fold Plot", style={'textAlign': 'center', 'marginTop': 50}),
    dcc.Graph(id='mse-plot')
])

# Define the callback to enable/disable the K input field
@app.callback(
    Output('k-value-input', 'disabled'),
    Input('validation-dropdown', 'value')
)
def toggle_k_input(selected_method):
    return selected_method != 'k_fold'

# Define the callback function to update dataset, output, and plot
@app.callback(
    [Output('output-container', 'children'),
     Output('regression-plot', 'src'),
     Output('mse-plot', 'figure')],
    [Input('validation-dropdown', 'value'),
     Input('k-value-input', 'value'),
     Input('num-samples-input', 'value'),
     Input('noise-input', 'value')]
)
def update_output(selected_method, k_value, n_samples, noise):
    try:
        # Validate inputs
        if n_samples is None or n_samples < 10:
            raise ValueError("Number of samples must be at least 10.")
        if noise is None or noise < 0:
            raise ValueError("Noise level cannot be negative.")

        # Generate a new dataset based on user input
        X, y = generate_data(n_samples=n_samples, noise=noise)

        if selected_method is None:
            return "", 'data:image/png;base64,{}'.format(plot_regression_line(X, y)), {}

        model = LinearRegression()
        
        if selected_method == 'same_dataset':
            mse = train_test_same_dataset(X, y, model)
            plot = plot_regression_line(X, y, model)
            return f"Mean Squared Error (Train/Test on Same Dataset): {mse:.2f}", 'data:image/png;base64,{}'.format(plot), {}
        elif selected_method == 'train_test_split':
            mse = train_test_split_method(X, y, model)
            plot = plot_regression_line(X, y, model)
            return f"Mean Squared Error (Train/Test Split): {mse:.2f}", 'data:image/png;base64,{}'.format(plot), {}
        elif selected_method == 'k_fold':
            mse_scores = []
            k_values = list(range(2, k_value + 1))
            for k in k_values:
                mse = k_fold_cross_validation(X, y, model, k=k)
                mse_scores.append(mse)

            # Create a plot for MSE vs K
            mse_fig = {
                'data': [{
                    'x': k_values,
                    'y': mse_scores,
                    'type': 'line',
                    'name': 'MSE vs K-Fold'
                }],
                'layout': {
                    'title': 'MSE vs K-Fold Cross-Validation',
                    'xaxis': {'title': 'K-Fold'},
                    'yaxis': {'title': 'Mean Squared Error'},
                }
            }

            plot = plot_regression_line(X, y, model)
            return f"Mean Squared Error (K-Fold Cross-Validation with k={k_value}): {mse_scores[-1]:.2f}", 'data:image/png;base64,{}'.format(plot), mse_fig
    
    except Exception as e:
        # Return an error message and plot the initial global dataset if an error occurs
        error_message = f"Error: {str(e)}"
        plot = plot_regression_line(initial_X, initial_y)  # Use the initial global dataset
        return error_message, 'data:image/png;base64,{}'.format(plot), {}

# Define the functions for each validation technique
def train_test_same_dataset(X, y, model):
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse

def train_test_split_method(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def k_fold_cross_validation(X, y, model, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
    
    average_mse = np.mean(mse_scores)
    return average_mse

# Run the app.
if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 8000))
    app.run_server(debug=False)#, host='0.0.0.0', port=port)
