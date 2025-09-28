# csc172-data-cleaning-CORPUZ

# Data Cleaning with AI Support

## Student Information
- Name: Joseph Jr. Q. Corpuz
- Course Year: BSCS 4
- Date: 2025 - 09 - 27

## Dataset
- Source: [\[Kaggle/UCI link\]](https://www.kaggle.com/c/titanic/data?select=train.csv)
- Name: Titanic Dataset

## Issues found
- Missing values: Age - 177 missing , 
    Cabin - 687 missing, 
    Embarked - 2 missing
- Duplicates: Nothing found
- Inconsistencies: Name has titles - Ms., Mrs., Dr.
    Sex - consistent but stored on strings
    Different Data types in Ticket and Cabin coulmns


## Cleaning steps
1. Missing values: Dropped Cabin column, Guessed age based on  the meidan grouped from title, Filled Embarked with 's', Extracted title, standardized it and made new column for it.
2. Duplicates: checked if any passenger have duplicates and drops them if found.
3. Inconsistencies:  lowered Sex and Embarked for consistency.
4. Outliers: detected 512 max in the Fare column, kept it as is even though it might skew results as the extreme fare might be due to other added amenities. 

## AI prompts used
- Prompt 1: "In the names of the passengers there are multiple people with Mr. or Ms. Am i able to standardize it?"
- Generated:

### ✅ Step-by-Step: How to Standardize Titles

#### 1. **Extract the title** (the word before the dot `.`)
```python
# Extract title using regex
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
```

> This pulls out `"Mr"`, `"Miss"`, `"Mlle"`, `"Ms"`, `"Dr"`, etc.

#### 2. **Map equivalent titles to standard forms**
```python
# Group rare titles
df['Title'] = df['Title'].replace([
    'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

# Standardize female titles
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')  # All = unmarried woman
df['Title'] = df['Title'].replace('Mme', 'Mrs')            # French for Mrs
```

#### 3. **(Optional) View results**
```python
print(df['Title'].value_counts())



### ✅ Final Code (Copy-Paste Ready)
# Extract and standardize passenger titles
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Group rare titles
df['Title'] = df['Title'].replace([
    'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

# Standardize female variants
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
```


--Prompt 2: The Age column has a lot of missing values. Is it possible to make an educated guess based on other columns?

Generated: 

---

### ✅ Best Approach: **Impute Age by Group Medians**
Use **`Title` + `Pclass`** to estimate age more accurately.

#### Step 1: **Create the `Title` column** (if you haven’t already)
```python
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['Title'] = df['Title'].replace([
    'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
```

#### Step 2: **Calculate median Age by Title + Pclass**
```python
# Create a mapping of (Title, Pclass) → median Age
age_medians = df.groupby(['Title', 'Pclass'])['Age'].median().to_dict()

# Example: age_medians[('Master', 3)] = 4.0
```

#### Step 3: **Fill missing Age values using the mapping**
```python
def impute_age(row):
    if pd.isnull(row['Age']):
        return age_medians.get((row['Title'], row['Pclass']), df['Age'].median())
    return row['Age']

df['Age'] = df.apply(impute_age, axis=1)
```

---
| Passenger | Raw Median Age | Group-Based Estimate |
|----------|----------------|----------------------|
| Boy with `"Master"` in 3rd class | 28 (overall median) | **~5 years old** ✅ |
| Young woman `"Miss"` in 1st class | 28 | **~30 years old** ✅ |
| Adult man `"Mr"` in 3rd class | 28 | **~25 years old** ✅ |

> 📊 **Result**: More realistic ages → better model performance!

---

### 💡 Alternative: Use Only `Title` (Simpler)
If you want a simpler approach:
```python
# Median age by Title only
title_age_map = df.groupby('Title')['Age'].median().to_dict()
df['Age'] = df['Age'].fillna(df['Title'].map(title_age_map))
```

Typical medians:
- `Master`: **4.0** (boys)
- `Miss`: **21.0** (young women)
- `Mrs`: **35.0** (married women)
- `Mr`: **30.0** (adult men)
- `Rare`: **45.0** (officers/nobility)

---

### 📊 Validate Your Imputation
Check if it makes sense:
```python
# Before imputation
print("Missing Age before:", df['Age'].isnull().sum())

# After imputation
print("Missing Age after:", df['Age'].isnull().sum())

# Check median Age by Title
print("\nMedian Age by Title:")
print(df.groupby('Title')['Age'].median())
```

Prompt 3: Given the context of the data cleaning instructions given to me, it says here to include before and after of shape. What does it mean by shape?

Generated: 
In the context of data cleaning, **"shape"** refers to the **dimensions of your dataset**, specifically:

- **Number of rows** (observations or records)  
- **Number of columns** (variables or features)

In Python (especially with pandas DataFrames), you can check the shape using the `.shape` attribute:

```python
df.shape
# Returns a tuple like (1000, 5) meaning 1000 rows and 5 columns
```

So when your instructions say **"include before and after of shape,"** it means you should:

1. **Before cleaning**: Report the shape of the original/raw dataset.  
2. **After cleaning**: Report the shape of the dataset after you’ve applied your cleaning steps (e.g., removing duplicates, dropping missing values, filtering rows, etc.).

**Example:**
```python
print("Shape before cleaning:", df.shape)
# ... your cleaning steps ...
print("Shape after cleaning:", df.shape)
```

This helps show how your cleaning process affected the size of the dataset—e.g., how many rows or columns were removed or added.


---

## Results
- Rows before: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
- Rows after: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Embarked,Title

Video: link
        