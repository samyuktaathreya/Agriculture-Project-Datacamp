import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

#check for missing values
print(crops.isna().sum().sort_values())

#check for crop types
print(crops['crop'].unique())

#split the data
X = crops.drop('crop', axis = 1).values
y = crops['crop']
X_train, X_test, y_train, y_test = train_test_split(X, y)

#evaluate feature performance
features_dict = {}
best_predictive_feature = {}
for feature in ['N', 'P', 'K', 'ph']:
    log_reg = LogisticRegression()
    multi_class = 'multinomial'
    # Use boolean indexing to select the feature column
    feature_index = crops.columns.get_loc(feature)
    log_reg.fit(X_train[:, feature_index].reshape(-1, 1), y_train)
    
    y_pred = log_reg.predict(X_test[:, feature_index].reshape(-1, 1))
    
    feature_performance = metrics.f1_score(y_test, y_pred, average = 'weighted')
    features_dict[feature] = feature_performance

#store best performing feature in single key dictionary
best_predictive_feature = {'K': features_dict['K']}