# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

from sklearn.model_selection import train_test_split
from eda_pipeline import run_eda   # EDA pipeline


# ============================================================
# 2. LOAD DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'us_perm_visas.csv')

df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)


# ============================================================
# 3. SELECT IMPORTANT RAW COLUMNS (ORIGINAL DATASET)
# ============================================================

important_cols = [
    'case_number',
    'case_received_date',
    'decision_date',
    'case_status',
    'class_of_admission',
    'country_of_citizenship',
    'job_info_job_title',
    'job_info_education',
    'agent_state',
    'agent_city'
]

df = df[important_cols]


# ============================================================
# 4. DATA CLEANING & TARGET CREATION
# ============================================================

df['case_received_date'] = pd.to_datetime(df['case_received_date'])
df['decision_date'] = pd.to_datetime(df['decision_date'])

df = df.dropna(subset=['case_received_date'])

# ---------------- TARGET VARIABLE ----------------
df['processing_days'] = (
    df['decision_date'] - df['case_received_date']
).dt.days

print(df['processing_days'].describe())

# Handle missing categorical values
df['class_of_admission'].fillna('Unknown_Class', inplace=True)
df['country_of_citizenship'].fillna(df['country_of_citizenship'].mode()[0], inplace=True)
df['job_info_job_title'].fillna(df['job_info_job_title'].mode()[0], inplace=True)
df['job_info_education'].fillna('Unknown', inplace=True)
df['agent_city'].fillna('Unknown_City', inplace=True)
df['agent_state'].fillna('Unknown_State', inplace=True)


# ============================================================
# 5. CORRECT TRAIN–TEST SPLIT (X / y SEPARATION)
# ============================================================

y = df['processing_days']        # Target
X = df.drop(columns=['processing_days'])  # Features

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)


# ============================================================
# 6. FEATURE ENGINEERING – JOB FEATURES
# ============================================================

def job_group(title):
    t = str(title).lower()
    if re.search(r'engineer|developer|software|architect|programmer|computer', t):
        return 'Engineering'
    if re.search(r'analyst|data', t):
        return 'Analyst'
    if re.search(r'scientist|research|lab|chemist|bio|pharma', t):
        return 'Science'
    if re.search(r'manager|director|vp|lead|supervisor', t):
        return 'Management'
    if re.search(r'technician|tech', t):
        return 'Technician'
    if re.search(r'nurse|medical|dental|therap', t):
        return 'Healthcare'
    if re.search(r'teacher|professor|instructor', t):
        return 'Education'
    return 'Other'


# Job category
X_train['job_category'] = X_train['job_info_job_title'].apply(job_group)
X_test['job_category'] = X_test['job_info_job_title'].apply(job_group)

# One-hot encoding
job_dummies_train = pd.get_dummies(X_train['job_category'], prefix='job', dtype=int)
job_dummies_test = pd.get_dummies(X_test['job_category'], prefix='job', dtype=int)

job_dummies_test = job_dummies_test.reindex(
    columns=job_dummies_train.columns, fill_value=0
)

X_train = pd.concat([X_train, job_dummies_train], axis=1)
X_test = pd.concat([X_test, job_dummies_test], axis=1)

# Job title frequency (train-only stats)
job_freq = X_train['job_info_job_title'].value_counts()
X_train['job_title_freq'] = X_train['job_info_job_title'].map(job_freq)
X_test['job_title_freq'] = X_test['job_info_job_title'].map(job_freq).fillna(0)


# ============================================================
# 7. FEATURE ENGINEERING – TARGET ENCODING (LEAKAGE SAFE)
# ============================================================

global_mean = y_train.mean()

# Agent state target encoding
state_tgt = y_train.groupby(X_train['agent_state']).mean()
X_train['agent_state_tgt'] = X_train['agent_state'].map(state_tgt)
X_test['agent_state_tgt'] = X_test['agent_state'].map(state_tgt)

X_train['agent_state_tgt'].fillna(global_mean, inplace=True)
X_test['agent_state_tgt'].fillna(global_mean, inplace=True)

# Country target encoding
country_tgt = y_train.groupby(X_train['country_of_citizenship']).mean()
X_train['citizenship_tgt'] = X_train['country_of_citizenship'].map(country_tgt)
X_test['citizenship_tgt'] = X_test['country_of_citizenship'].map(country_tgt)

X_train['citizenship_tgt'].fillna(global_mean, inplace=True)
X_test['citizenship_tgt'].fillna(global_mean, inplace=True)

# Visa class target encoding
visa_tgt = y_train.groupby(X_train['class_of_admission']).mean()
X_train['visa_type'] = X_train['class_of_admission'].map(visa_tgt)
X_test['visa_type'] = X_test['class_of_admission'].map(visa_tgt)

X_train['visa_type'].fillna(global_mean, inplace=True)
X_test['visa_type'].fillna(global_mean, inplace=True)


# ============================================================
# 8. FEATURE ENGINEERING – EDUCATION FEATURES
# ============================================================

def clean_education_text(x):
    x = str(x).lower()
    x = re.sub(r"'s|’s", "", x)
    x = re.sub(r"[^a-z ]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def simplify_education(x):
    x = clean_education_text(x)
    if 'doctorate' in x: return 'Doctorate'
    if 'master' in x: return 'Master'
    if 'bachelor' in x: return 'Bachelor'
    if 'associate' in x: return 'Associate'
    if 'high school' in x: return 'High_School'
    if x == '' or 'unknown' in x: return 'Unknown'
    return 'Other'


X_train['edu_simple'] = X_train['job_info_education'].apply(simplify_education)
X_test['edu_simple'] = X_test['job_info_education'].apply(simplify_education)

edu_dummies_train = pd.get_dummies(X_train['edu_simple'], dtype=int)
edu_dummies_test = pd.get_dummies(X_test['edu_simple'], dtype=int)

edu_dummies_test = edu_dummies_test.reindex(
    columns=edu_dummies_train.columns, fill_value=0
)

X_train = pd.concat([X_train, edu_dummies_train], axis=1)
X_test = pd.concat([X_test, edu_dummies_test], axis=1)


# ============================================================
# 9. RUN EDA (ON FULL DATA FOR INSIGHTS ONLY)
# ============================================================

run_eda(df, X_train.assign(processing_days=y_train))


# ============================================================
# 10. FINAL CLEANUP – MODEL-READY DATA
# ============================================================

drop_cols = [
    'case_number', 'case_received_date', 'decision_date', 'case_status',
    'job_info_job_title', 'job_info_education',
    'agent_city', 'agent_state',
    'country_of_citizenship', 'class_of_admission',
    'job_category', 'edu_simple'
]

X_train.drop(columns=drop_cols, inplace=True)
X_test.drop(columns=drop_cols, inplace=True)

print("Final X_train shape:", X_train.shape)
print("Final X_test shape:", X_test.shape)
