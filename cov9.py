# -*- coding: utf-8 -*-
"""
Created on Thu May 30 02:24:04 2024


@author: Alemu Derseh; Dereje Merrie; Gusha Belako; Kehabtimer Shiferaw
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_excel("COVID_worldwide.xlsx")  # Use the relative path

# Data cleaning

# Replace negative 'cases' and 'deaths' with their absolute values
df['cases'] = df['cases'].abs()
df['deaths'] = df['deaths'].abs()

# Ensure the data is sorted by date for each country
df['dateRep'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')
df.sort_values(by=['countriesAndTerritories', 'dateRep'], inplace=True)

# ReCalculate the 14-day cumulative number of COVID-19 cases per 100,000 population

def calculate_cumulative_cases_per_100000(group):
    group['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'] = (
        group['cases'].rolling(window=14, min_periods=1).sum() / group['popData2019'] * 100000
    )
    return group

# Apply the function to each group of countries
df = df.groupby('countriesAndTerritories', group_keys=False).apply(calculate_cumulative_cases_per_100000)

# Save the updated dataframe to the original Excel file
df.to_excel("COVID_worldwide.xlsx", index=False)

# Set the title of the Streamlit app
st.title("Global COVID-19 Data Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# Filter data based on user selection
selected_continent = st.sidebar.multiselect('Select Continent', df['continentExp'].unique())
if selected_continent:
    filtered_countries = df[df['continentExp'].isin(selected_continent)]['countriesAndTerritories'].unique()
    selected_country = st.sidebar.multiselect('Select Country', filtered_countries)
else:
    selected_country = st.sidebar.multiselect('Select Country', df['countriesAndTerritories'].unique())

selected_year = st.sidebar.multiselect('Select Year', df['year'].unique())
if selected_year:
    filtered_months = df[df['year'].isin(selected_year)]['dateRep'].dt.strftime('%b-%Y').unique()
    selected_month = st.sidebar.multiselect('Select Month', filtered_months)
else:
    selected_month = st.sidebar.multiselect('Select Month', df['dateRep'].dt.strftime('%b-%Y').unique())

# Apply filters
filtered_df = df.copy()
if selected_continent:
    filtered_df = filtered_df[filtered_df['continentExp'].isin(selected_continent)]
if selected_country:
    filtered_df = filtered_df[filtered_df['countriesAndTerritories'].isin(selected_country)]
if selected_year:
    filtered_df = filtered_df[filtered_df['year'].isin(selected_year)]
if selected_month:
    filtered_df = filtered_df[filtered_df['dateRep'].dt.strftime('%b-%Y').isin(selected_month)]

# Display raw data checkbox
if st.checkbox("Display the raw data"):
    # Display the data frame
    st.subheader("COVID-19 Data")
    st.write(filtered_df)

# Calculate mean, standard deviation, and sum values for the first table
mean_cases = filtered_df['cases'].mean()
std_cases = filtered_df['cases'].std()
mean_deaths = filtered_df['deaths'].mean()
std_deaths = filtered_df['deaths'].std()
sum_cases = filtered_df['cases'].sum()
sum_deaths = filtered_df['deaths'].sum()

# Helper function to format large numbers consistently
def human_format(num, pos=None):
    if num >= 1e6:
        return f'{num*1e-6:.2f}M'
    elif num >= 1e3:
        return f'{num*1e-3:.2f}K'
    else:
        return str(int(num))

# Display mean (+/- SD) and sum data boxes only for the first table
col1, col2 = st.columns(2)

with col1:
    st.info(f"Mean Cases: {mean_cases:.2f} (+/- {std_cases:.2f})")
    st.info(f"Total Cases: {human_format(sum_cases)}")

with col2:
    st.info(f"Mean Deaths: {mean_deaths:.2f} (+/- {std_deaths:.2f})")
    st.info(f"Total Deaths: {human_format(sum_deaths)}")

# Visualization 1: COVID-19 Cases by Country and Deaths by Country (Side by Side)
st.subheader("Number of COVID-19 Cases and Deaths by Country")

# Calculate top countries cases and deaths
top_countries_cases = filtered_df.groupby('countriesAndTerritories')['cases'].sum().nlargest(20)
top_countries_deaths = filtered_df.groupby('countriesAndTerritories')['deaths'].sum().nlargest(20)

# Plot top countries cases and deaths
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
top_countries_cases.plot(kind='bar', ax=ax1, color='skyblue')
top_countries_deaths.plot(kind='bar', ax=ax2, color='salmon')

# Set titles and labels
ax1.set_title("Top 20 Countries by COVID-19 Cases")
ax1.set_ylabel("Total Cases (in 100,000)")
ax1.set_xlabel("Countries")
ax1.set_yticklabels([f'{int(y/1e5):,}' for y in ax1.get_yticks()])
ax2.set_title("Top 20 Countries by COVID-19 Deaths")
ax2.set_ylabel("Total Deaths")
ax2.set_xlabel("Countries")
fig.suptitle("Figure 1: Top 20 Countries by COVID-19 Cases and Deaths", fontsize=16)

# Display the visualization
st.pyplot(fig)

# Visualization 3: Weekly Statistics and Visualization 4: Rate of Increase (Side by Side)
st.subheader("Weekly COVID-19 Cases and Rate of Increase")

filtered_df['week'] = filtered_df['dateRep'].dt.isocalendar().week
weekly_cases = filtered_df.groupby('week')['cases'].sum()
filtered_df_sorted = filtered_df.sort_values(by='dateRep')
filtered_df_sorted['cumulative_cases'] = filtered_df_sorted.groupby('countriesAndTerritories')['cases'].cumsum()
filtered_df_sorted['rate_of_increase'] = filtered_df_sorted['cumulative_cases'].pct_change().fillna(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

weekly_cases.plot(ax=ax1, label='Cases', color='blue')
ax1.set_title("Weekly COVID-19 Cases")
ax1.set_ylabel("Total Cases (in 100,000)")
ax1.set_xlabel("Week Number")
ax1.set_yticklabels([f'{int(y/1e5):,}' for y in ax1.get_yticks()])

filtered_df_sorted.groupby('dateRep')['rate_of_increase'].mean().plot(ax=ax2, color='red')
ax2.set_title("Average Rate of Increase of COVID-19 Cases")
ax2.set_ylabel("Rate of Increase")
ax2.set_xlabel("Date")

fig.suptitle("Figure 3: Weekly COVID-19 Cases and Rate of Increase", fontsize=16)
st.pyplot(fig)

# Visualization 5: Cases and Deaths per 100,000 Population and Scatterplot
st.subheader("COVID-19 Cases and Deaths per 100,000 Population")
filtered_df['cases_per_100k'] = filtered_df['cases'] * 100000 / filtered_df['popData2019']
filtered_df['deaths_per_100k'] = filtered_df['deaths'] * 100000 / filtered_df['popData2019']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

filtered_df.groupby('dateRep')['cases_per_100k'].mean().plot(ax=ax1, label='Cases', color='purple')
filtered_df.groupby('dateRep')['deaths_per_100k'].mean().plot(ax=ax1, label='Deaths', color='orange')
ax1.set_title("COVID-19 Cases and Deaths per 100,000 Population")
ax1.set_ylabel("Rate per 100,000")
ax1.set_xlabel("Date")
ax1.legend()

sns.scatterplot(x='cases_per_100k', y='deaths_per_100k', data=filtered_df, ax=ax2)
sns.regplot(x='cases_per_100k', y='deaths_per_100k', data=filtered_df, scatter=False, ax=ax2, color='red', label='Trend line')
ax2.set_title("Scatterplot of Cases vs Deaths per 100,000 Population")
ax2.set_xlabel("Cases per 100,000")
ax2.set_ylabel("Deaths per 100,000")
ax2.legend()

fig.suptitle("Figure 4: COVID-19 Cases and Deaths per 100,000 Population and Scatterplot", fontsize=16)
st.pyplot(fig)

# Visualization 6: COVID-19 Cases and Deaths per 100,000 Population by Continent and Monthly CFR Trends (Side by Side)
st.subheader("COVID-19 Cases and Deaths by Continent and Trends of COVID-19 Case Fatality Rate over Months")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Group by continent and sum cases and deaths, then sort by cases
df_continent = df.groupby('continentExp')[['cases', 'deaths']].sum()
df_continent['total'] = df_continent['cases'] + df_continent['deaths']
df_continent = df_continent.sort_values(by='cases', ascending=True)

# Plot horizontal bar graph for cases and deaths
df_continent[['cases', 'deaths']].plot(kind='barh', stacked=True, ax=ax1, color=['skyblue', 'salmon'])
ax1.set_title("COVID-19 Cases and Deaths by Continent")
ax1.set_xlabel("Rate per 100,000")
ax1.set_ylabel("Continents")
ax1.legend(['Cases', 'Deaths'], loc='center', bbox_to_anchor=(0.4, 0.2))  # Move legend to the right
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))  # Format x-axis labels with commas

# Add data labels to the bars
for p in ax1.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax1.annotate(f'{int(width)}', (x + width/2, y + height/2), ha='center', va='center')

filtered_df['month'] = filtered_df['dateRep'].dt.to_period('M')
monthly_cfr = filtered_df.groupby('month')['deaths'].sum() / filtered_df.groupby('month')['cases'].sum() * 100

monthly_cfr.plot(ax=ax2, marker='o', color='red')
ax2.set_title("Trends of COVID-19 Case Fatality Rate over Months")
ax2.set_ylabel("Case Fatality Rate (%)")
ax2.set_xlabel("Month")
ax2.set_xticklabels([label.strftime('%b %Y') for label in monthly_cfr.index.to_timestamp()], rotation=45)
for idx, val in monthly_cfr.items():
    ax2.text(idx.to_timestamp(), val, f'{val:.2f}', ha='center', va='bottom')

fig.suptitle("Figure 5: COVID-19 Cases and Deaths by Continent and Trends of COVID-19 Case Fatality Rate over Months", fontsize=16)
st.pyplot(fig)

# Visualization 7: 14 Days Cumulative COVID-19 Cases per 100,000 Population
st.subheader("Figure 6: 14 Days Cumulative COVID-19 Cases per 100,000 Population")

# Filter for a specific country for better visualization (optional)
selected_country_for_cumulative = st.selectbox('Select Country for Cumulative Cases', filtered_df['countriesAndTerritories'].unique())
country_df = filtered_df[filtered_df['countriesAndTerritories'] == selected_country_for_cumulative]

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(data=country_df, x='dateRep', y='Cumulative_number_for_14_days_of_COVID-19_cases_per_100000', marker='o', color='blue', ax=ax)

# Formatting the plot
ax.set_title('14 Days Cumulative COVID-19 Cases per 100,000 Population')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative number for 14 days of COVID-19 cases per 100,000')
ax.grid(True)

# Show the plot in Streamlit
st.pyplot(fig)
