# ==================================================================================================================================================
# Author                    : Dr. Marcos H. Cárdenas Mancilla
# E-mail                    : marcos.cardenas.m@usach.cl
# Date of creation          : 2024-11-16
# Licence                   : AGPL V3
# Copyright (c) 2024 Marcos H. Cárdenas Mancilla.
# ==================================================================================================================================================
# Descripción de Translog_XML_Analyzer_PY:
# Este script de Python analiza datos de tiempo de respuesta (RT) extraídos de archivos XML que fueron generados Translog II (versión 2.0) (Carl, 2012). 
# El código procesa múltiples archivos XML para el análisis comparativo de RTs entre diferentes grupos de variables intrasujeto e intratarea y sus interacciones.
# Características del pipeline metodologógico:
# 1. extracción información sobre participantes, niveles de experiencia, texto de la tarea de traducción, tipo de evento y acciones realizadas.
# 2. cálculo de los tiempos de respuesta (RT) como la diferencia entre eventos consecutivos.
# 3. análisis estadísticos automatizados i.e., pruebas de normalidad (Shapiro-Wilk y D’Agostino-Pearson), Kruskal-Wallis para comparar grupos,
# y Dunn's post-hoc para comparaciones por pares. 
# 4. visualización del resultados del análisis descriptivo (p. ej. tablas, barras, boxplots, mapas de calor).
# 5. cálculo de tamaños de efecto a partir del análisis inferencial (p. ej. correlación biserial de rango) para determinar los efectos significativos
# que permitan comprender las relaciones entre RT y las variables independientes (p. ej. experiencia en traducción, tipo de acción y texto).
# ==================================================================================================================================================

# Load libraries
import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from scipy.stats import shapiro
import numpy as np
from scipy.stats import boxcox
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import warnings

# Function to process XML files
def process_task_xml(file_path, participant, experience, text):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    events = []
    for event in root.findall(".//Events/*"):
        time = int(event.attrib.get('Time'))
        event_type = event.tag.lower()  # Lowercase variable labels
        action = event.attrib.get('Type', event_type.lower())
        value = event.attrib.get('Value', '')

        events.append({
            'participant': participant.lower(),
            'experience': experience.lower(),
            'text': text.upper(),
            'time': time,
            'type': event_type,
            'action': action,
            'value': value
        })
    return pd.DataFrame(events)

# Directory containing XML files
dir_path = r'translog'

# Extract grouping variables from file names and process files
dfs = []
for file_name in os.listdir(dir_path):
    if file_name.endswith('.xml'):
        file_path = os.path.join(dir_path, file_name)
        # Extract participant, experience, and text from file name
        parts = file_name.split('.')[0].split('_')
        if len(parts) == 3:
            participant, experience, text = parts
            dfs.append(process_task_xml(file_path, participant, experience, text))
        else:
            print(f"Skipping file with unexpected pattern: {file_name}")

# Combine data from all files
df = pd.concat(dfs, ignore_index=True)

# Calculate Pause and RT
df['pause'] = df.groupby(['participant', 'experience', 'text'])['time'].diff().fillna(0)
df['rt'] = df['pause']

# Calculate statistics grouped by participant, experience, text, and action type
rt_by_action = df.groupby(['participant', 'experience', 'text', 'action'])['rt'].agg(['mean', 'std', 'median', 'count']).reset_index()

# Create metadata summary
metadata = [
    {
        'participant': row['participant'],
        'experience': row['experience'],
        'text': row['text'],
        'variable': row['action'],
        'description': f"Events of type '{row['action']}' for participant '{row['participant']}', experience '{row['experience']}', and text '{row['text']}'",
        'unit': 'milliseconds',
        'scale': 'interval',
        'count': int(row['count']),
        'mean rt': round(row['mean'], 2),
        'std dev rt': round(row['std'], 2),
        'median rt': round(row['median'], 2)
    } for _, row in rt_by_action.iterrows()
]

metadata_df = pd.DataFrame(metadata)
print("Metadata Summary Table:")
print(metadata_df)

# Visualization: Mean RT by action type for each grouping
plt.figure(figsize=(15, 6))
sns.barplot(
    x='action', y='mean', hue='text', data=rt_by_action,
    errorbar='se', palette='colorblind', capsize=0.1
)
plt.xlabel('Action Type')
plt.ylabel('Mean Response Time (ms)')
plt.title('Mean Response Times by Action Type across Texts')
plt.legend(title='Text')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Visualization: Event counts by participant, experience, and text
event_counts = df.groupby(['participant', 'experience', 'text', 'action']).size().reset_index(name='count')
plt.figure(figsize=(15, 6))
sns.barplot(
    x='action', y='count', hue='text', data=event_counts,
    palette='colorblind', errorbar=None
)
plt.xlabel('Action Type')
plt.ylabel('Count')
plt.title('Event Counts by Action Type across Texts')
plt.legend(title='Text')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate statistics grouped by participant, experience, text, and action type
rt_by_action = df.groupby(['participant', 'experience', 'text', 'action'])['rt'].agg(['mean', 'std', 'median', 'count']).reset_index()

# Create a new DataFrame with the desired size and structure
new_df = rt_by_action.head(269)  # Adjust this line to filter rows as needed
print("New DataFrame:")
print(new_df)
print("New DataFrame Shape:", new_df.shape)

metadata_df = pd.DataFrame(metadata)
print("Metadata Summary Table:")
print(metadata_df)

# RT by Action Types with error bars
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df, x='action', y='rt', hue='action', errorbar='se', palette='colorblind', 
    dodge=False, legend=False, capsize=0.1
)
plt.xlabel('Action Type')
plt.ylabel('Response Time (ms)')
plt.title('RT by Action Types with Standard Error')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# RT by Text with error bars
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df, x='text', y='rt', hue='text', errorbar='se', palette='colorblind',
    dodge=False, legend=False, capsize=0.1
)
plt.xlabel('Text')
plt.ylabel('Response Time (ms)')
plt.title('RT by Text with Standard Error')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# RT by Experience with error bars
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df, x='experience', y='rt', hue='experience', errorbar='se',
    palette='colorblind', dodge=False, legend=False, capsize=0.1
)
plt.xlabel('Experience')
plt.ylabel('Response Time (ms)')
plt.title('RT by Experience with Standard Error')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Enhancing the metadata table to include sublevels for 'type' and 'action' variables
metadata_table = pd.DataFrame({
    "Variable": ["participant", "experience", "text", "time", "type", "action", "value", "pause", "rt"],
    "Description": [
        "Name of the participant extracted from the file label.",
        "Experience level or type, either 'lat' or 'pei'.",
        "The text associated with the task (e.g., 'A', 'B', or 'C').",
        "Time of the event in milliseconds.",
        "Type of event recorded. Sublevels include: 'key' (keyboard events) and 'mouse' (mouse events).",
        "Specific action taken. Sublevels include: 'insert', 'delete', 'navi' (navigation), and 'mouse'.",
        "Value associated with the event (if any).",
        "Pause time calculated as the difference between consecutive events.",
        "Response Time (RT) calculated as the pause time between events."
    ],
    "Unit": [
        "String", "String", "String", "Milliseconds", "String", "String", "String", "Milliseconds", "Milliseconds"
    ],
    "Scale": [
        "Nominal", "Nominal", "Nominal", "Interval", "Nominal", "Nominal", "Nominal", "Interval", "Interval"
    ]
})

# Displaying the enhanced metadata table
print("Enhanced Metadata Table:")
print(metadata_table)

# Save the metadata table to a CSV file
metadata_table.to_csv("enhanced_metadata_table.csv", index=False)
print("The metadata table has been saved as 'enhanced_metadata_table.csv'.")

# Function to process XML files
def process_task_xml(file_path, participant, experience, text):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    events = []
    for event in root.findall(".//Events/*"):
        time = int(event.attrib.get('Time'))
        event_type = event.tag.lower()  # Standardize variable names to lowercase
        action = event.attrib.get('Type', event_type.lower())
        value = event.attrib.get('Value', '')

        events.append({
            'participant': participant.lower(),
            'experience': experience.lower(),
            'text': text.upper(),
            'time': time,
            'type': event_type,
            'action': action,
            'value': value
        })
    return pd.DataFrame(events)

# Process all XML files
dfs = []
for file_name in os.listdir(dir_path):
    if file_name.endswith('.xml'):
        file_path = os.path.join(dir_path, file_name)
        # Extract participant, experience, and text from file name
        participant, experience, text = file_name.split('.')[0].split('_')
        dfs.append(process_task_xml(file_path, participant, experience, text))

# Combine data from all files into a single DataFrame
df = pd.concat(dfs, ignore_index=True)

# Calculate pause and RT (Response Time)
df['pause'] = df.groupby(['participant', 'experience', 'text'])['time'].diff().fillna(0)
df['rt'] = df['pause']

# Function to check normality for small and large samples
def check_normality_mixed(df, group_by):
    results = []
    grouped_data = df.groupby(group_by)
    for group, data in grouped_data:
        rt_values = data['rt'].dropna()
        if len(rt_values) > 2:
            if len(rt_values) >= 20:  # Large sample: Use D’Agostino-Pearson
                stat, p_value = normaltest(rt_values)
                test_name = "D'Agostino-Pearson"
            else:  # Small sample: Use Shapiro-Wilk
                stat, p_value = shapiro(rt_values)
                test_name = "Shapiro-Wilk"
            results.append({
                group_by: group,
                "Test": test_name,
                "Statistic": stat,
                "p-value": p_value,
                "Normality": "Yes" if p_value > 0.05 else "No"
            })
    return pd.DataFrame(results)

# Perform normality checks with mixed testing approach
normality_by_experience = check_normality_mixed(df, 'experience')
normality_by_type = check_normality_mixed(df, 'type')
normality_by_action = check_normality_mixed(df, 'action')

# Print results
print("Normality Results by Experience:")
print(normality_by_experience.to_string(index=False))

print("\nNormality Results by Type:")
print(normality_by_type.to_string(index=False))

print("\nNormality Results by Action:")
print(normality_by_action.to_string(index=False))

# Save normality results to CSV
normality_by_experience.to_csv("normality_by_experience_mixed.csv", index=False)
normality_by_type.to_csv("normality_by_type_mixed.csv", index=False)
normality_by_action.to_csv("normality_by_action_mixed.csv")

print("\nNormality results have been saved as:")
print("- 'normality_by_experience_mixed.csv'")
print("- 'normality_by_type_mixed.csv'")
print("- 'normality_by_action_mixed.csv'")

# Select the column to transform (e.g., 'rt' column)
rt_data = df['rt'].dropna()  # Remove NaN values

# Ensure all values are positive
if (rt_data <= 0).any():
    print("Data contains non-positive values. Adding a constant to make it positive.")
    rt_data += abs(rt_data.min()) + 1

# Apply Box-Cox transformation
transformed_data, lambda_opt = boxcox(rt_data)

# Print optimal lambda
print(f"Optimal Lambda for Box-Cox Transformation: {lambda_opt}")

# Compare before and after transformation
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(rt_data, bins=30, color='blue', alpha=0.7, label='Original Data')
plt.title("Original Data Distribution")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(transformed_data, bins=30, color='green', alpha=0.7, label='Box-Cox Transformed Data')
plt.title("Box-Cox Transformed Data Distribution")
plt.legend()

plt.tight_layout()
plt.show()

# Function to compare means using Kruskal-Wallis test
def compare_means_kruskal(df, group_by):
    results = []
    grouped_data = df.groupby(group_by)
    for group, data in grouped_data:
        rt_values = data['rt'].dropna()
        if len(rt_values) > 2:  # Kruskal-Wallis requires at least 3 observations per group
            groups = [grp['rt'].dropna().values for _, grp in grouped_data]
            stat, p_value = kruskal(*groups)
            results.append({
                "Grouping Variable": group_by,
                "Group": group,
                "Kruskal-Wallis Statistic": stat,
                "p-value": p_value,
                "Significant Difference": "Yes" if p_value < 0.05 else "No"
            })
    return pd.DataFrame(results)

# Perform Kruskal-Wallis test for experience, action, and text
kw_experience = compare_means_kruskal(df, 'experience')
kw_action = compare_means_kruskal(df, 'action')
kw_text = compare_means_kruskal(df, 'text')

# Save Kruskal-Wallis results to CSV files
kw_experience.to_csv("kruskal_results_by_experience.csv", index=False)
kw_action.to_csv("kruskal_results_by_action.csv", index=False)
kw_text.to_csv("kruskal_results_by_text.csv", index=False)

print("Kruskal-Wallis results have been saved as:")
print("- 'kruskal_results_by_experience.csv'")
print("- 'kruskal_results_by_action.csv'")
print("- 'kruskal_results_by_text.csv'")

# Print the Kruskal-Wallis results for each group
print("Kruskal-Wallis Results by Experience:")
print(kw_experience.to_string(index=False))

print("\nKruskal-Wallis Results by Action:")
print(kw_action.to_string(index=False))

print("\nKruskal-Wallis Results by Text:")
print(kw_text.to_string(index=False))

# 1. Post-hoc Analysis using Dunn's Test
# Perform Dunn's test for pairwise comparisons within each grouping variable
posthoc_experience = posthoc_dunn(df, val_col='rt', group_col='experience', p_adjust='bonferroni')
posthoc_action = posthoc_dunn(df, val_col='rt', group_col='action', p_adjust='bonferroni')
posthoc_text = posthoc_dunn(df, val_col='rt', group_col='text', p_adjust='bonferroni')

# Save Dunn's Test results to CSV
posthoc_experience.to_csv("dunn_results_by_experience.csv")
posthoc_action.to_csv("dunn_results_by_action.csv")
posthoc_text.to_csv("dunn_results_by_text.csv")

# 2. Visualization of RT Distributions (Boxplots)
plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='experience', y='rt', palette='colorblind')
plt.title("Response Time (RT) Distribution by Experience")
plt.xlabel("Experience")
plt.ylabel("Response Time (ms)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='action', y='rt', palette='colorblind')
plt.title("Response Time (RT) Distribution by Action")
plt.xlabel("Action")
plt.ylabel("Response Time (ms)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='text', y='rt', palette='colorblind')
plt.title("Response Time (RT) Distribution by Text")
plt.xlabel("Text")
plt.ylabel("Response Time (ms)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 3. Combined Analysis
# Heatmaps of Dunn's post-hoc results
plt.figure(figsize=(10, 8))
sns.heatmap(posthoc_experience, annot=True, cmap='coolwarm', fmt=".3f")
plt.title("Dunn's Post-Hoc Test for Experience")
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(posthoc_action, annot=True, cmap='coolwarm', fmt=".3f")
plt.title("Dunn's Post-Hoc Test for Action")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(posthoc_text, annot=True, cmap='coolwarm', fmt=".3f")
plt.title("Dunn's Post-Hoc Test for Text")
plt.show()

print("Post-hoc analysis and visualizations complete. Results saved as:")
print("- 'dunn_results_by_experience.csv'")
print("- 'dunn_results_by_action.csv'")
print("- 'dunn_results_by_text.csv'")

# Display results
print("\nDunn's Test Results:")
print("Experience Pairwise Comparisons:")
print(posthoc_experience)

print("\nAction Pairwise Comparisons:")
print(posthoc_action)

print("\nText Pairwise Comparisons:")
print(posthoc_text)

# Function to perform a Kruskal-Wallis test for interaction effects
def interaction_kruskal(df, group1, group2):
    results = []
    grouped_data = df.groupby([group1, group2])
    for (g1, g2), data in grouped_data:
        rt_values = data['rt'].dropna()
        if len(rt_values) > 2:  # Kruskal-Wallis requires at least 3 observations per group
            results.append({
                group1: g1,
                group2: g2,
                "Median RT": rt_values.median(),
                "RT Count": len(rt_values)
            })
    return pd.DataFrame(results)

# Prepare data for pairwise Kruskal-Wallis analysis
interaction_results = interaction_kruskal(df, 'experience', 'text')

# Kruskal-Wallis test on combined groups
grouped_rt = [
    group['rt'].dropna().values
    for _, group in df.groupby(['experience', 'text'])
    if len(group['rt'].dropna()) > 2
]
kw_stat, kw_p_value = kruskal(*grouped_rt)

# Save the results
interaction_results.to_csv("interaction_kruskal_results.csv", index=False)

# Display Kruskal-Wallis test result
print("Kruskal-Wallis Test for Interaction Between Experience and Text:")
print(f"Statistic: {kw_stat}, p-value: {kw_p_value}")

# Visualization of interaction effects
plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='text', y='rt', hue='experience', palette='colorblind')
plt.title(f"Interaction Between Experience and Text on RT (Kruskal-Wallis p-value: {kw_p_value:.4f})")
plt.xlabel("Text")
plt.ylabel("Response Time (ms)")
plt.legend(title="Experience")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Save interaction results to CSV
try:
    interaction_results.to_csv("interaction_kruskal_results.csv", index=False)
    print("Saved: 'interaction_kruskal_results.csv'")
except Exception as e:
    print(f"Error saving interaction results: {e}")

# Print the Kruskal-Wallis test summary
print("\nKruskal-Wallis Test Summary:")
print(f"Statistic: {kw_stat:.4f}, p-value: {kw_p_value:.4f}")
print("Significant Interaction Effect" if kw_p_value < 0.05 else "No Significant Interaction Effect")

from scikit_posthocs import posthoc_dunn

# Perform Dunn's test for pairwise comparisons within experience groups
contrasts_experience_text = []
for exp in df['experience'].unique():
    subset = df[df['experience'] == exp]
    contrast = posthoc_dunn(subset, val_col='rt', group_col='text', p_adjust='bonferroni')
    contrasts_experience_text.append((exp, contrast))

# Save and display results
for exp, contrast in contrasts_experience_text:
    filename = f"dunn_contrast_experience_{exp}_text.csv"
    contrast.to_csv(filename)
    print(f"Saved contrasts for Experience = {exp}: {filename}")

# Perform Dunn's test for pairwise comparisons within text groups
contrasts_text_experience = []
for txt in df['text'].unique():
    subset = df[df['text'] == txt]
    contrast = posthoc_dunn(subset, val_col='rt', group_col='experience', p_adjust='bonferroni')
    contrasts_text_experience.append((txt, contrast))

# Save and display results
for txt, contrast in contrasts_text_experience:
    filename = f"dunn_contrast_text_{txt}_experience.csv"
    contrast.to_csv(filename)
    print(f"Saved contrasts for Text = {txt}: {filename}")
# Interaction plot
plt.figure(figsize=(12, 6))
sns.pointplot(data=df, x='text', y='rt', hue='experience', ci='sd', palette='colorblind')
plt.title("Interaction Effects of Experience and Text on RT")
plt.xlabel("Text")
plt.ylabel("Response Time (ms)")
plt.legend(title="Experience")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Print interaction contrasts for Experience × Text
print("Pairwise Contrasts for Experience × Text:")

# Experience contrasts across Text levels
for exp, contrast in contrasts_experience_text:
    print(f"\nContrasts for Experience = {exp} across Text levels:")
    print(contrast.to_string())

# Text contrasts across Experience levels
for txt, contrast in contrasts_text_experience:
    print(f"\nContrasts for Text = {txt} across Experience levels:")
    print(contrast.to_string())



# Function to calculate effect size (Cohen's d or rank-biserial correlation)
def calculate_effect_size(group1, group2):
    u_stat, _ = mannwhitneyu(group1, group2, alternative='two-sided')
    n1, n2 = len(group1), len(group2)
    rank_biserial_corr = (2 * u_stat) / (n1 * n2) - 1  # Rank-biserial correlation
    return rank_biserial_corr

# Prepare effect size calculations for significant pairwise contrasts
effect_sizes = []

# Experience × Text contrasts
for exp, contrast in contrasts_experience_text:
    for text1 in contrast.columns:
        for text2 in contrast.index:
            if text1 != text2 and contrast.loc[text2, text1] < 0.05:  # Significant
                group1 = df[(df['experience'] == exp) & (df['text'] == text1)]['rt'].dropna()
                group2 = df[(df['experience'] == exp) & (df['text'] == text2)]['rt'].dropna()
                effect_size = calculate_effect_size(group1, group2)
                effect_sizes.append({
                    "Experience": exp,
                    "Text Pair": f"{text1} vs {text2}",
                    "Effect Size (Rank-Biserial)": effect_size
                })

# Text × Experience contrasts
for txt, contrast in contrasts_text_experience:
    for exp1 in contrast.columns:
        for exp2 in contrast.index:
            if exp1 != exp2 and contrast.loc[exp2, exp1] < 0.05:  # Significant
                group1 = df[(df['text'] == txt) & (df['experience'] == exp1)]['rt'].dropna()
                group2 = df[(df['text'] == txt) & (df['experience'] == exp2)]['rt'].dropna()
                effect_size = calculate_effect_size(group1, group2)
                effect_sizes.append({
                    "Text": txt,
                    "Experience Pair": f"{exp1} vs {exp2}",
                    "Effect Size (Rank-Biserial)": effect_size
                })

# Convert effect sizes to DataFrame
effect_sizes_df = pd.DataFrame(effect_sizes)

# Save and display the results
effect_sizes_df.to_csv("effect_sizes_significant_contrasts.csv", index=False)

print("Effect Sizes for Significant Pairwise Comparisons:")
print(effect_sizes_df.to_string(index=False))

# Updated function to handle empty groups
action_effect_sizes = []

# Group by action types and perform pairwise comparisons
for action in df['action'].unique():
    action_data = df[df['action'] == action]
    for exp in action_data['experience'].unique():
        subset = action_data[action_data['experience'] == exp]
        
        # Ensure there are at least 3 unique texts with sufficient data
        valid_texts = subset['text'].value_counts()
        valid_texts = valid_texts[valid_texts > 2].index
        
        if len(valid_texts) >= 2:  # At least 2 groups required for comparisons
            contrast = posthoc_dunn(subset[subset['text'].isin(valid_texts)],
                                    val_col='rt', group_col='text', p_adjust='bonferroni')
            for text1 in contrast.columns:
                for text2 in contrast.index:
                    if text1 != text2 and contrast.loc[text2, text1] < 0.05:  # Significant
                        group1 = subset[(subset['text'] == text1)]['rt'].dropna()
                        group2 = subset[(subset['text'] == text2)]['rt'].dropna()
                        if len(group1) > 0 and len(group2) > 0:
                            effect_size = calculate_effect_size(group1, group2)
                            action_effect_sizes.append({
                                "Action Type": action,
                                "Experience": exp,
                                "Text Pair": f"{text1} vs {text2}",
                                "Effect Size (Rank-Biserial)": effect_size
                            })

# Convert to DataFrame
action_effect_sizes_df = pd.DataFrame(action_effect_sizes)

# Save and display the results
try:
    action_effect_sizes_df.to_csv("effect_sizes_by_action.csv", index=False)
    print("Saved: 'effect_sizes_by_action.csv'")
except Exception as e:
    print(f"Error saving effect sizes: {e}")

# Display results
print("Effect Sizes for Significant Contrasts by Action Types:")
print(action_effect_sizes_df.to_string(index=False))