import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from sklearn.neural_network import MLPRegressor


def automl():

    def build_model(df, model):
        
        X = df.iloc[:, :-1]  # Using all column except for the last column as X
        Y = df.iloc[:, -1]  # Selecting the last column as Y

        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=(100-split_size)/100)

        st.markdown('**1.2. Data splits**')
        st.write('Training set')
        st.info(X_train.shape)
        st.write('Test set')
        st.info(X_test.shape)

        st.markdown('**1.3. Variable details**:')
        st.write('X variable')
        st.info(list(X.columns))
        st.write('Y variable')
        st.info(Y.name)

        if str(model) == 'Random Forest Regressor':

            rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
                                       random_state=parameter_random_state,
                                       max_features=parameter_max_features,
                                       criterion=parameter_criterion,
                                       min_samples_split=parameter_min_samples_split,
                                       min_samples_leaf=parameter_min_samples_leaf,
                                       bootstrap=parameter_bootstrap,
                                       oob_score=parameter_oob_score,
                                       n_jobs=parameter_n_jobs)
            rf.fit(X_train, Y_train)

        elif str(model) == 'Multi Layer Perceptron':

            hidden_layers = [parameter_neurons] * parameter_layers
            hidden_layers = tuple(hidden_layers)

            rf = MLPRegressor(hidden_layer_sizes=hidden_layers,
                              activation=parameter_activation,
                              solver=parameter_solver,
                              learning_rate_init=parameter_learning_rate,
                              max_iter=200,)

            rf.fit(X_train, Y_train)

        elif str(model) == 'Gradient Boosting Regressor':

            rf = GradientBoostingRegressor(loss=parameter_loss,
                                           learning_rate=parameter_learning_rate,
                                           n_estimators=parameter_n_estimators)

            rf.fit(X_train, Y_train)

        st.subheader('2. Model Performance')

        st.markdown('**2.1. Training set**')
        Y_pred_train = rf.predict(X_train)
        st.write('Coefficient of determination ($R^2$):')
        st.info(r2_score(Y_train, Y_pred_train))

        st.write('Error (MSE or MAE):')
        st.info(mean_squared_error(Y_train, Y_pred_train))

        st.markdown('**2.2. Test set**')
        Y_pred_test = rf.predict(X_test)
        st.write('Coefficient of determination ($R^2$):')
        st.info(r2_score(Y_test, Y_pred_test))

        st.write('Error (MSE or MAE):')
        st.info(mean_squared_error(Y_test, Y_pred_test))

        st.subheader('3. Model Parameters')
        st.write(rf.get_params())

        st.subheader('4. Predict on your own data')

        with st.sidebar.header('4. Predict on your own data'):

            uploaded_file_test = st.sidebar.file_uploader(
                "Upload your test CSV file", type=["csv"])
            st.sidebar.markdown("""
            [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
            """)

        if uploaded_file_test is not None:

            df = pd.read_csv(uploaded_file_test)

            Y_predicted = rf.predict(df)
            st.write("Predicted values")
            st.info(Y_predicted)
            
            st.write('Coefficient of determination ($R^2$):')
            st.info(r2_score(Y_train, Y_predicted))

            st.write('Error (MSE or MAE):')
            st.info(mean_squared_error(Y_test, Y_predicted))
        
        else:

            st.info('Awaiting for CSV file to be uploaded.')

    #---------------------------------#

    st.markdown('''
    ***EasyML***

    This is the **machine learning** section of the EasyML App

    Built by [Nalin Nagar](https://github.com/NNDEV1/)

    ''')

    st.subheader('0. Choose model:')
    st.info('Choose your model wisely, you will be making predictions using this model!')
    model = st.selectbox("Which model would you like to use?",
                         ('Multi Layer Perceptron', 'Random Forest Regressor', 'Gradient Boosting Regressor'))

    with st.sidebar.header('1. Upload your CSV data'):

        uploaded_file = st.sidebar.file_uploader(
            "Upload your input CSV file", type=["csv"])
        st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider(
            'Data split ratio (% for Training Set)', 10, 90, 80, 5)

    if model == 'Random Forest Regressor':

        with st.sidebar.subheader('2.1. Learning Parameters'):

            parameter_n_estimators = st.sidebar.slider(
                'Number of estimators (n_estimators)', 0, 1000, 100, 100)
            parameter_max_features = st.sidebar.select_slider(
                'Max features (max_features)', options=['auto', 'sqrt', 'log2'])
            parameter_min_samples_split = st.sidebar.slider(
                'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
            parameter_min_samples_leaf = st.sidebar.slider(
                'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    elif model == 'Multi Layer Perceptron':

        with st.sidebar.subheader('2.1. Learning Parameters'):

            parameter_layers = st.sidebar.slider(
                'Number of layers (hidden_layer_sizes)', 1, 5, 3, 1)
            parameter_neurons = st.sidebar.slider(
                'Number of neurons (hidden_layer_sizes)', 5, 100, 50, 10)
            parameter_activation = st.sidebar.select_slider(
                'Activation function for the hidden layer (activation)', options=['identity', 'logistic', 'tanh', 'relu'])
            parameter_solver = st.sidebar.select_slider(
                'Solver for weight optimization (solver)', options=['lbfgs', 'sgd', 'adam'])
            parameter_learning_rate = st.sidebar.slider(
                'The initial learning rate used (learning_rate_init)', 0.00001, 0.1, 0.001, 0.001)

    elif model == 'Gradient Boosting Regressor':

        with st.sidebar.subheader('2.1. Learning Parameters'):

            parameter_loss = st.sidebar.select_slider(
                'Loss function to be optimized (loss)', options=['huber', 'quantile'])
            parameter_n_estimators = st.sidebar.slider(
                'Number of estimators (n_estimators)', 0, 1000, 100, 100)
            parameter_learning_rate = st.sidebar.slider(
                'The initial learning rate used (learning rate) *Note there is a trade-off between n_estimatiors and learning rate', 0.00001, 0.1, 0.001, 0.001)

    with st.sidebar.subheader('2.2. General Parameters'):

        parameter_random_state = st.sidebar.slider(
            'Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.sidebar.select_slider(
            'Performance measure (criterion)', options=['mse', 'mae'])
        parameter_bootstrap = st.sidebar.select_slider(
            'Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.sidebar.select_slider(
            'Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
        parameter_n_jobs = st.sidebar.select_slider(
            'Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    #---------------------------------#
    # Main panel

    # Displays the dataset
    st.subheader('1. Dataset')

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)
            st.markdown('**1.1. Glimpse of dataset**')
            st.write(df)

            build_model(df, model)

        except:
            
            st.warning("An error has occured with your data, please check your data and press \"rerun\" from the top left menu when you are ready")

    else:

        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):

            # Boston housing dataset
            boston = load_boston()
            X = pd.DataFrame(boston.data, columns=boston.feature_names)
            Y = pd.Series(boston.target, name='response')
            df = pd.concat([X, Y], axis=1)

            st.markdown('The Boston housing dataset is used as the example.')
            st.write(df.head(5))

            build_model(df, model)
        
