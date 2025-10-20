#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Property Value Prediction Script

This script loads a model from MLflow, predicts property values,
and saves a database with the required columns.

Created on: April 26, 2025
Author: Based on work by @roman
"""

import os
import logging
import warnings
import joblib
import argparse

import pandas as pd
import numpy as np
import yaml

import mlflow
import xgboost as xgb
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column",
    module="mlflow.types.utils"
)


# Utility functions
def get_configs(filepath):
    with open(filepath, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            config_model = config['model']
            recaster_mappers = config['categorical_recasters']
            mlflow_config = config['mlflow']
        except yaml.YAMLError as exc:
            print(exc)
    return config_model, recaster_mappers, mlflow_config


def get_target(df, config):
    # read
    target_name = list(config['target'].keys())[0]

    # if ppsm2 is in target_name, then we need to calculate the target
    if 'ppsm' in target_name:
        if 'built' in target_name:
            y = df['price'] / df['built_area']
        elif 'saleable' in target_name:
            y = df['price'] / df['saleable_area']
        elif 'land' in target_name:
            y = df['price'] / df['land_area']
        else:
            raise ValueError('Target is not supported')
    elif 'price' in target_name:
        y = df['price']
    else:
        raise ValueError('Target is not supported')
    return y.copy()


def get_target_value(y, transformer):
    return transformer.inverse_transform_target(y)


def get_predictions(model, data, transformer):
    return get_target_value(model.predict(data), transformer)


def read_data(file, config):
    # read data
    df_work = pd.read_parquet(file)

    # filter data
    n_rows = config.get('data').get('n_rows')
    if n_rows:
        if n_rows < df_work.shape[0]:
            df_work = df_work.sample(n_rows, random_state=42).reset_index(drop=True)

    # set observation_id as index
    df_work = df_work.set_index('observation_id')

    # new features
    first_date_obs = df_work['valuation_date'].min()
    df_work = (
        df_work
        .assign(
            # quarters since appraisal
            quarters_since_first_appraisal=lambda x: (
                x['valuation_date'] - first_date_obs
                ).dt.days / (30.4 * 3),
            # recategorizations
            is_new=lambda x:
                np.where(x['age_in_months'].le(1), 'new', 'used'),
            conservacion_recat=lambda x:
                x['conservation_status_id'].replace({7: 3.5}).astype(int),
            is_renovated=lambda x:
                np.where(x['conservation_status_id'].ge(7), 'renovated', 'not_renovated'),
            is_apartment=lambda x:
                np.where(x['property_type_id'].ge(4), 1, 0),
            has_elevator=lambda x:
                np.where(x['elevator_service_id'].eq(1), 1, 0),
        )
        .assign(
            is_new=lambda x: x['is_new'].astype('category'),
        )
    )

    # set target
    features_name = list(config['features'].keys())
    X = df_work.loc[:, features_name].copy()
    y = get_target(df_work, config)

    return X, y


# Feature engineering functions
def feature_engineering(X, y, config, recaster_mapper=None):
    transformers = {}  # Dictionary to store all fitted transformers

    # Step 1: Check and handle missing values
    missing_values = X.isnull().sum().sum()
    if missing_values > 0:
        logger.warning(f"{missing_values} missing values detected in the dataset")

    # Step 2: Apply recaster mapping if provided
    if recaster_mapper:
        try:
            X = X.replace(recaster_mapper)
            logger.info("Applied recaster mapping successfully")
        except Exception as e:
            logger.error(f"Error applying recaster mapping: {str(e)}")
            raise ValueError(f"Failed to apply recaster mapping: {str(e)}")

    # Step 3: Handle categorical columns
    try:
        cols_categorical = list(set(
            X.select_dtypes(include=['string', 'category', 'object']).columns.tolist()
        ))
        X[cols_categorical] = X[cols_categorical].astype('category')
        transformers['categorical'] = cols_categorical
        logger.info(f"Processed {len(cols_categorical)} categorical columns")
    except Exception as e:
        logger.error(f"Error processing categorical columns: {str(e)}")
        raise ValueError(f"Failed to process categorical columns: {str(e)}")

    # Step 4: Apply feature transformations based on config
    valid_transformations = ['identity', 'log', 'sqrt', 'standardize', 'boxcox']
    X_transformed = X.copy()

    # Validate and group transformations
    transformation_groups = {trans_type: [] for trans_type in valid_transformations}
    for feature, transformation in config['features'].items():
        if transformation not in valid_transformations:
            raise ValueError(
                f"Unsupported transformation type for feature '{feature}': "
                f"'{transformation}'. Supported types are: "
                f"{', '.join(valid_transformations)}"
            )
        if feature not in X.columns:
            logger.warning(
                f"Feature '{feature}' not found in the dataset. "
                "Skipping transformation."
            )
            continue
        if transformation in ['log', 'boxcox'] and (
                X[feature].min() <= 0 or pd.isna(X[feature]).any()):
            logger.warning(
                f"Feature '{feature}' contains values <= 0 or NaN. "
                "Using standardize instead."
            )
            transformation = 'standardize'
        if transformation == 'sqrt' and (
                X[feature].min() < 0 or pd.isna(X[feature]).any()):
            logger.warning(
                f"Feature '{feature}' contains values < 0 or NaN. "
                "Using standardize instead."
            )
            transformation = 'standardize'
        transformation_groups[transformation].append(feature)

    # Apply transformations
    for transformation, features in transformation_groups.items():
        if not features or transformation == 'identity':
            logger.info(f"Skipping '{transformation}' transformation for {len(features)} features")
            continue
        try:
            logger.info(f"Applying '{transformation}' transformation to {len(features)} features")
            X_transformed[features], transformer = apply_transformation(
                X[features], transformation, feature_name=features
            )
            if transformer is not None:
                transformers[transformation] = transformer
        except Exception as e:
            logger.error(
                f"Error applying {transformation} transformation to features "
                f"{features}: {str(e)}"
            )
            raise ValueError(f"Failed to apply {transformation} transformation: {str(e)}")

    logger.info("Applied transformations to features successfully")

    # Step 5: Transform target variable if specified
    if 'target' in config:
        target_name = list(config['target'].keys())[0]
        target_transformation = config['target'][target_name]
        if target_transformation not in valid_transformations:
            raise ValueError(
                f"Unsupported transformation type for target: '{target_transformation}'. "
                f"Supported types are: {', '.join(valid_transformations)}"
            )
        try:
            if target_transformation in ['log', 'boxcox', 'sqrt'] and (
                    np.min(y) <= 0 or np.isnan(y).any()):
                logger.warning(
                    f"Target contains values <= 0 or NaN. Using standardize instead."
                )
                target_transformation = 'standardize'
            y_transformed, target_transformer = apply_transformation(
                y.to_frame(), target_transformation, feature_name=target_name
            )
            if isinstance(y_transformed, np.ndarray):
                y_transformed = pd.Series(y_transformed.flatten(), index=y.index)
            else:
                y_transformed = pd.Series(y_transformed.to_numpy().flatten(), index=y.index)
            if target_transformer is not None:
                transformers["target"] = target_transformer
            logger.info(f"Applied '{target_transformation}' transformation to target variable")
        except Exception as e:
            logger.error(f"Error transforming target variable: {str(e)}")
            raise ValueError(f"Failed to transform target variable: {str(e)}")
    else:
        logger.warning("No target transformation specified in config")

    return X_transformed, y_transformed, transformers


def apply_transformation(data, transformation_type, feature_name=None):
    name_info = f" for '{feature_name}'" if feature_name else ""

    if transformation_type == 'identity':
        return data, None

    if transformation_type == 'log':
        transformed_data = np.log(data)
        transformer = StandardScaler()
        transformed_data = transformer.fit_transform(transformed_data)
        return transformed_data, transformer

    if transformation_type == 'sqrt':
        transformed_data = np.sqrt(data)
        transformer = StandardScaler()
        transformed_data = transformer.fit_transform(transformed_data)
        return transformed_data, transformer

    if transformation_type == 'standardize':
        transformer = StandardScaler()
        transformed_data = transformer.fit_transform(data)
        return transformed_data, transformer

    if transformation_type == 'boxcox':
        transformer = PowerTransformer(method='box-cox', standardize=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            transformed_data = transformer.fit_transform(data)
        return transformed_data, transformer

    raise ValueError(f"Unsupported transformation type{name_info}: '{transformation_type}'")


def save_transformers(transformers, filepath):
    try:
        joblib.dump(transformers, filepath)
        logger.info(f"Saved transformers to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving transformers: {str(e)}")
        return False


def load_transformers(filepath):
    try:
        transformers = joblib.load(filepath)
        logger.info(f"Loaded transformers from {filepath}")
        return transformers
    except Exception as e:
        logger.error(f"Error loading transformers: {str(e)}")
        raise ValueError(f"Failed to load transformers: {str(e)}")


def transform_features(X, transformers):
    X_transformed = X.copy()

    # Apply categorical transformations
    if 'categorical' in transformers:
        cols_categorical = transformers['categorical']
        present_cols = [col for col in cols_categorical if col in X.columns]
        X[present_cols] = X[present_cols].astype('category')

    for transformation_type, features in transformers.items():
        if transformation_type == 'categorical' and len(features) > 0:
            cols_categorical = transformers['categorical']
            present_cols = [col for col in cols_categorical if col in X.columns]
            X[present_cols] = X[present_cols].astype('category')

        elif transformation_type == 'log' and len(features) > 0:
            X_transformed[features] = np.log(X[features])
            transformer = transformers['log']
            X_transformed[features] = transformer.transform(X_transformed[features])

        elif transformation_type == 'sqrt' and len(features) > 0:
            X_transformed[features] = np.sqrt(X[features])
            transformer = transformers['sqrt']
            X_transformed[features] = transformer.transform(X_transformed[features])

        elif transformation_type == 'standardize' and len(features) > 0:
            transformer = transformers['standardize']
            X_transformed[features] = transformer.transform(X[features])

        elif transformation_type == 'boxcox' and len(features) > 0:
            transformer = transformers['boxcox']
            X_transformed[features] = transformer.transform(X[features])

    return X_transformed


def inverse_transform_target(y_pred, transformation_type, transformers):
    if "target" not in transformers or transformers["target"] is None:
        return y_pred

    try:
        # inverse transform target
        y_pred_reshaped = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
        inverse_transformed = transformers["target"].inverse_transform(y_pred_reshaped)

        # case for log and sqrt (because is not a pipeline)
        if transformation_type == 'log':
            inverse_transformed = np.exp(inverse_transformed)
        elif transformation_type == 'sqrt':
            inverse_transformed = inverse_transformed ** 2

        if inverse_transformed.shape[1] == 1:
            inverse_transformed = inverse_transformed.flatten()

        return inverse_transformed
    except Exception as e:
        logger.error(f"Error inverse transforming target: {str(e)}")
        raise ValueError(f"Failed to inverse transform target: {str(e)}")


class FeatureTransformer:
    def __init__(self, config, recaster_mapper=None):
        self.config = config
        self.recaster_mapper = recaster_mapper
        self.transformers_ = None
        self.target_transformation_ = list(config.get('target').values())[0]

    def fit(self, X, y=None):
        """Fit the transformer on the training data."""
        _, _, self.transformers_ = feature_engineering(X, y, self.config, self.recaster_mapper)
        return self

    def transform(self, X):
        """Transform the features."""
        return transform_features(X, self.transformers_)

    def fit_transform(self, X, y=None):
        """Fit the transformer and transform the features."""
        X_transformed, y_transformed, self.transformers_ = feature_engineering(
            X, y, self.config, self.recaster_mapper
        )
        return X_transformed, y_transformed

    def get_transformers(self):
        """Get the fitted transformers."""
        return self.transformers_

    def save_transformers(self, filepath):
        """Save the fitted transformers to a file."""
        return save_transformers(self.transformers_, filepath)

    def load_transformers(self, filepath):
        """Load transformers from a file."""
        self.transformers_ = load_transformers(filepath)
        return self

    def inverse_transform_target(self, y_pred):
        """Inverse transform the target variable."""
        return inverse_transform_target(
            y_pred, self.target_transformation_, self.transformers_
        )


# Split data and prepare XGBoost data format
def split_randomly_data(X, y, config, categorical_features=None):
    """
    Split data into train, validation, and test sets based on the provided configuration.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target dataset.
        config (dict): Configuration dictionary containing:
            - train_size (float): Proportion of data for training.
            - stratify (str or None): Column name for stratification.
            - validation_size (float or None): Proportion of training data for validation.
        categorical_features (list or None): List of categorical feature names.

    Returns:
        dict: A dictionary containing the train, validation, and test sets as DataFrames/Series
              and corresponding DMatrix objects for XGBoost.
    """
    # For XGBoost, we need to encode categorical features
    X_encoded = X.copy()
    
    # Convert categorical features to numeric using one-hot encoding
    X_encoded[categorical_features] = X_encoded[categorical_features].astype('category')

    # Split train and test
    stratify_col = X[config['stratify']] if config.get('stratify') else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        train_size=config['train_size'],
        random_state=42,
        stratify=stratify_col
    )

    # Split validation set if specified
    val_percentage = config.get('validation_size')
    if val_percentage and val_percentage > 0:
        stratify_col_train = X_train[config['stratify']] if (
            config.get('stratify') and config['stratify'] in X_train
        ) else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_percentage,
            random_state=42,
            stratify=stratify_col_train
        )
    else:
        X_val, y_val = None, None

    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True) if X_val is not None else None

    return {
        'train': {
            'dmatrix': dtrain,
            'X': X_train,
            'y': y_train
        },
        'validation': {
            'dmatrix': dval,
            'X': X_val,
            'y': y_val
        },
        'test': {
            'dmatrix': dtest,
            'X': X_test,
            'y': y_test
        }
    }


def main(config_path, model_id):
    """Main function to run property value prediction"""

    # Get configuration
    config_model, recaster_mappers, mlflow_config = get_configs(config_path)

    # Silence warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Define paths
    output_path = f"../../data/processed/properties_predicted_{model_id}.parquet"

    DIR_DATA = '../../data'
    FILE_PROPERTIES = f"{DIR_DATA}/clean/properties_shif.parquet"

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"http://{mlflow_config['host']}:{mlflow_config['port']}")
    mlflow.set_experiment(mlflow_config['experiment_name'])

    # Load model from MLflow using only the model_id
    model_path = f"runs:/{model_id}/models"
    logger.info(f"Loading model from MLflow: {model_path}")
    model = mlflow.xgboost.load_model(model_path)
    logger.info("Model loaded successfully")

    # Read and prepare data
    logger.info("Reading data...")
    X, y = read_data(FILE_PROPERTIES, config_model)
    logger.info(f"Data read successfully. X shape: {X.shape}, y shape: {y.shape}")

    # Feature engineering
    logger.info("Applying feature engineering...")
    transformer = FeatureTransformer(config_model, recaster_mappers)
    X_transformed, y_transformed = transformer.fit_transform(X, y)  # not the best practice, but ok for now (!)
    logger.info(
        f"Feature engineering completed. "
        f"X shape: {X_transformed.shape}, y shape: {y_transformed.shape}"
    )

    # Split data
    logger.info("Splitting data into train-test sets...")
    pools = split_randomly_data(
        X_transformed,
        y_transformed,
        config_model['data'],
        categorical_features=transformer.get_transformers()['categorical']
    )

    # Free memory
    del X, y, X_transformed, y_transformed

    # Make predictions
    logger.info("Making predictions...")
    for label in pools.keys():
        logger.info(f"Predicting {label} set...")
        # Get original scale of y
        pools[label]['y_original'] = get_target_value(
            pools[label]['y'].values,
            transformer
        )

        # Predict
        pools[label]['predictions'] = get_predictions(
            model,
            pools[label]['dmatrix'],
            transformer
        )

    # Append predictions to each X
    logger.info("Appending predictions to datasets...")
    for label in pools.keys():
        dict_target = pools[label]

        # Original scale
        dict_target['X']['y_pred'] = dict_target['predictions']
        dict_target['X']['y'] = dict_target['y_original']

        # Fitted scale
        dict_target['X']['y_pred_fitted'] = model.predict(dict_target['dmatrix'])
        dict_target['X']['y_fitted'] = dict_target['y']

    # Combine all datasets
    logger.info("Combining datasets...")
    y_cols = ['y', 'y_pred', 'y_fitted', 'y_pred_fitted']
    df_properties_prices = pd.concat(
        [
            pools[label]['X'].assign(set=label).loc[:, y_cols + ['set']]
            for label in pools.keys()
        ],
        axis=0
    )

    # Get full property data
    logger.info("Reading property data...")
    df_properties = pd.read_parquet(FILE_PROPERTIES)
    df_properties = df_properties.set_index('observation_id')

    # Join data
    logger.info("Joining data...")
    df_properties = df_properties_prices.join(
        df_properties,
        on='observation_id',
        how='left'
    )

    # Save results
    logger.info(f"Saving results to {output_path}...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_properties.reset_index().to_parquet(output_path, index=False)
    logger.info(f"Results saved successfully. Shape: {df_properties.shape}")

    return df_properties


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict property values using MLflow model')
    parser.add_argument(
        '--config',
        type=str,
        default='config_xgb.yaml',
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--model-id',
        type=str,
        default="1ec464a33bdc436295dbb3c05e3c6ecb",
        help='MLflow model ID (just the run ID, not the full path)'
    )

    args = parser.parse_args()

    main(args.config, args.model_id)
