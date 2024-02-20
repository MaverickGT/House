import kfp
import kfp.components as comp
import requests
import kfp.dsl as dsl

def manage_outliers(df, column, method='IQR'):
    """
    Manage outliers in a dataframe column.

    Parameters:
    df (DataFrame): The Pandas DataFrame.
    column (str): The column name where to look for and manage outliers.
    method (str): Method to use for managing outliers. Default is 'IQR' (Interquartile Range).

    Returns:
    DataFrame: DataFrame with outliers managed.
    """
    df = df.copy()

    # Interquartile Range Method
    if method == 'IQR':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df

def prepare_data():
    import pandas as pd
    print("---- Inside prepare_data component ----")
    # Load dataset
    house_data = pd.read_csv("house.csv")
    house_data = house_data.dropna()
    # List of numerical columns to check for outliers
    numerical_columns = ['SqFt', 'Bedrooms', 'Bathrooms', 'Offers', 'Price']

    # Managing outliers for each numerical column
    for col in numerical_columns:
        house_data = manage_outliers(house_data, col, method='IQR')
        
    house_data = pd.get_dummies(house_data)
    house_data.to_csv(f'data/final_house.csv', index=False)
    print("\n ---- data csv is saved to PV location final_house.csv ----")

def train_test_split():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    print("---- Inside train_test_split component ----")
    final_data = pd.read_csv(f'data/final_house.csv')
    target_column = 'Price'
    X = final_data.loc[:, final_data.columns != target_column]
    y = final_data.loc[:, final_data.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,stratify = y, random_state=40)
    
    np.save(f'data/X_train.npy', X_train)
    np.save(f'data/X_test.npy', X_test)
    np.save(f'data/y_train.npy', y_train)
    np.save(f'data/y_test.npy', y_test)
    
    print("\n---- X_train ----")
    print("\n")
    print(X_train)
    
    print("\n---- X_test ----")
    print("\n")
    print(X_test)
    
    print("\n---- y_train ----")
    print("\n")
    print(y_train)
    
    print("\n---- y_test ----")
    print("\n")
    print(y_test)

def training_basic_classifier():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    print("---- Inside training_basic_classifier component ----")
    
    X_train = np.load(f'data/X_train.npy',allow_pickle=True)
    y_train = np.load(f'data/y_train.npy',allow_pickle=True)
    
    classifier = LinearRegression()
    classifier.fit(X_train,y_train)
    import pickle
    with open(f'data/model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    print("\n linear regression classifier is trained on house data and saved to location /data/model.pkl ----")

def predict_on_test_data():
    import pandas as pd
    import numpy as np
    import pickle
    print("---- Inside predict_on_test_data component ----")
    with open(f'data/model.pkl','rb') as f:
        linear_reg_model = pickle.load(f)
    X_test = np.load(f'data/X_test.npy',allow_pickle=True)
    y_pred = linear_reg_model.predict(X_test)
    np.save(f'data/y_pred.npy', y_pred)
    
    print("\n---- Predicted classes ----")
    print("\n")
    print(y_pred)

def predict_prob_on_test_data():
    import pandas as pd
    import numpy as np
    import pickle
    print("---- Inside predict_prob_on_test_data component ----")
    with open(f'data/model.pkl','rb') as f:
        linear_reg_model = pickle.load(f)
    X_test = np.load(f'data/X_test.npy',allow_pickle=True)
    y_pred_prob = linear_reg_model.predict_proba(X_test)
    np.save(f'data/y_pred_prob.npy', y_pred_prob)
    
    print("\n---- Predicted Probabilities ----")
    print("\n")
    print(y_pred_prob)

def get_metrics():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    from sklearn import metrics
    print("---- Inside get_metrics component ----")
    y_test = np.load(f'data/y_test.npy',allow_pickle=True)
    y_pred = np.load(f'data/y_pred.npy',allow_pickle=True)
    y_pred_prob = np.load(f'data/y_pred_prob.npy',allow_pickle=True)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    entropy = log_loss(y_test, y_pred_prob)
    
    y_test = np.load(f'data/y_test.npy',allow_pickle=True)
    y_pred = np.load(f'data/y_pred.npy',allow_pickle=True)
    print(metrics.classification_report(y_test, y_pred))

    print("\n Model Metrics:", {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)})

create_step_prepare_data = kfp.components.create_component_from_func(
    func=prepare_data,
    base_image='python:3.10',
    packages_to_install=['pandas==2.2.0','numpy==1.26.3']
)

create_step_train_test_split = kfp.components.create_component_from_func(
    func=train_test_split,
    base_image='python:3.10',
    packages_to_install=['pandas==2.2.0','numpy==1.26.3','scikit-learn==1.4.0']
)

create_step_training_basic_classifier = kfp.components.create_component_from_func(
    func=training_basic_classifier,
    base_image='python:3.10',
    packages_to_install=['pandas==2.2.0','numpy==1.26.3','scikit-learn==1.4.0']
)

create_step_predict_on_test_data = kfp.components.create_component_from_func(
    func=predict_on_test_data,
    base_image='python:3.10',
    packages_to_install=['pandas==2.2.0','numpy==1.26.3','scikit-learn==1.4.0']
)

create_step_predict_prob_on_test_data = kfp.components.create_component_from_func(
    func=predict_prob_on_test_data,
    base_image='python:3.10',
    packages_to_install=['pandas==2.2.0','numpy==1.26.3','scikit-learn==1.4.0']
)

create_step_get_metrics = kfp.components.create_component_from_func(
    func=get_metrics,
    base_image='python:3.10',
    packages_to_install=['pandas==2.2.0','numpy==1.26.3','scikit-learn==1.4.0']
)

# Define the pipeline
@dsl.pipeline(
   name='House price predicter Kubeflow Pipeline',
   description='A sample pipeline that performs House price prediction'
)
# Define parameters to be fed into pipeline
def house_predictor_pipeline(data_path: str):
    vop = dsl.VolumeOp(
    name="t-vol",
    resource_name="t-vol", 
    size="1Gi", 
    modes=dsl.VOLUME_MODE_RWO)
    
    prepare_data_task = create_step_prepare_data().add_pvolumes({data_path: vop.volume})
    train_test_split = create_step_train_test_split().add_pvolumes({data_path: vop.volume}).after(prepare_data_task)
    classifier_training = create_step_training_basic_classifier().add_pvolumes({data_path: vop.volume}).after(train_test_split)
    log_predicted_class = create_step_predict_on_test_data().add_pvolumes({data_path: vop.volume}).after(classifier_training)
    log_predicted_probabilities = create_step_predict_prob_on_test_data().add_pvolumes({data_path: vop.volume}).after(log_predicted_class)
    log_metrics_task = create_step_get_metrics().add_pvolumes({data_path: vop.volume}).after(log_predicted_probabilities)

    
    prepare_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_test_split.execution_options.caching_strategy.max_cache_staleness = "P0D"
    classifier_training.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_predicted_class.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_predicted_probabilities.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_metrics_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

kfp.compiler.Compiler().compile(
    pipeline_func=house_predictor_pipeline,
    package_path='house_predictor_pipeline.yaml')

client = kfp.Client()

DATA_PATH = '/data'

import datetime
print(datetime.datetime.now().date())


pipeline_func = house_predictor_pipeline
experiment_name = 'house_predictor_exp' +"_"+ str(datetime.datetime.now().date())
run_name = pipeline_func.__name__ + ' run'
namespace = "kubeflow"

arguments = {"data_path":DATA_PATH}

kfp.compiler.Compiler().compile(pipeline_func,  
  '{}.zip'.format(experiment_name))

run_result = client.create_run_from_pipeline_func(pipeline_func, 
                                                  experiment_name=experiment_name, 
                                                  run_name=run_name, 
                                                  arguments=arguments)

