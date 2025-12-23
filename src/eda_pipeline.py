import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_eda(df, train_df):

    print("🔍 Running Exploratory Data Analysis (EDA)...")

    # ==================================================
    # 1. Education vs Processing Days
    # ==================================================
    plt.figure(figsize=(10,5))
    sns.boxplot(
        x='job_info_education',
        y='processing_days',
        data=df,
        order=df.groupby('job_info_education')['processing_days']
              .mean()
              .sort_values()
              .index
    )
    plt.title("Visa Processing Days by Education Level")
    plt.xlabel("Education Level")
    plt.ylabel("Processing Days")
    plt.xticks(rotation=45)
    plt.show()

    edu_mean = (
        df.groupby('job_info_education')['processing_days']
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(8,4))
    edu_mean.plot(kind='bar')
    plt.title("Average Processing Days by Education Level")
    plt.ylabel("Average Processing Days")
    plt.show()

    # ==================================================
    # 2. Country of Citizenship vs Processing Days
    # ==================================================
    top_10_countries = (
        df['country_of_citizenship']
        .value_counts()
        .head(10)
        .index
    )

    df_top10 = df[df['country_of_citizenship'].isin(top_10_countries)]

    country_mean = (
        df_top10
        .groupby('country_of_citizenship')['processing_days']
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(10,5))
    country_mean.plot(kind='bar')
    plt.title("Average Visa Processing Days (Top 10 Countries)")
    plt.xlabel("Country")
    plt.ylabel("Average Processing Days")
    plt.xticks(rotation=45)
    plt.show()

    # ==================================================
    # 3. Class of Admission vs Processing Days
    # ==================================================
    # Select top 10 visa classes by count
    top_10_classes = (
    df['class_of_admission']
    .value_counts()
    .head(10)
    .index)

    df_top10_class = df[df['class_of_admission'].isin(top_10_classes)]

    class_mean = (
        df_top10_class
        .groupby('class_of_admission')['processing_days']
        .mean()
        .sort_values()
        .reset_index()
    )
    class_mean.columns = ['class_of_admission', 'avg_processing_days']

    plt.figure(figsize=(10,5))
    sns.barplot(
        data=class_mean,
        x='class_of_admission',
        y='avg_processing_days',
        palette='tab10'
    )
    plt.title("Average Visa Processing Days by Class of Admission (Top 10)")
    plt.xlabel("Class of Admission")
    plt.ylabel("Average Processing Days")
    plt.xticks(rotation=45)
    plt.show()

    # ==================================================
    # 4. Job Category vs Processing Days
    # ==================================================
    job_cat_mean = (
        train_df.groupby('job_category')['processing_days']
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(8,4))
    job_cat_mean.plot(kind='bar', color=plt.cm.Set2.colors)
    plt.title("Average Visa Processing Days by Job Category")
    plt.ylabel("Average Processing Days")
    plt.show()

    # ==================================================
    # 5. Month vs Processing Days (Seasonality)
    # ==================================================
    df['received_month_name'] = df['case_received_date'].dt.month_name()

    month_trend = (
        df.groupby('received_month_name')['processing_days']
        .mean()
    )

    plt.figure(figsize=(8,4))
    plt.plot(month_trend.index, month_trend.values, marker='o')
    plt.title("Seasonal Trend in Visa Processing Days")
    plt.xlabel("Month")
    plt.ylabel("Average Processing Days")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # ==================================================
    # 6. Agent State vs Processing Days
    # ==================================================
    top_10_states = (
        df['agent_state']
        .value_counts()
        .head(10)
        .index
    )

    df_top10_state = df[df['agent_state'].isin(top_10_states)]

    plt.figure(figsize=(10,5))
    sns.violinplot(
        data=df_top10_state,
        x='agent_state',
        y='processing_days',
        hue='agent_state',
        inner='quartile',
        palette='Set2',
        legend=False,
        linewidth=1.2
    )
    plt.title("Visa Processing Days by Agent State (Top 10)")
    plt.xlabel("Agent State")
    plt.ylabel("Processing Days")
    plt.xticks(rotation=45)
    plt.show()

    print("✅ EDA Completed Successfully")
