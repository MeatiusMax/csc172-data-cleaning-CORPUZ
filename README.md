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
Prompt 4: can you debug this code for me and explain mistakes?


import pandas as pd
import numpy as np

## 2. Load Dataset
df = pd.read_csv("../data/raw_dataset.csv")
df.head()

## 3. Initial Exploration
df.info()
df.describe()

#before cleaning
#sample rows
print(df.head())
print("Shape before Cleaning",df.shape)

## 4. Handle Missing Values
# Example: fill numeric NaNs with median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())
df.drop(columns=['Cabin'], inplace=True)

#AI promted to do this part
# Extract and standardize passenger titles
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Group rare titles
df['Title'] = df['Title'].replace([
    'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

# Standardize female variants
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Impute missing ages based on Title
title_age_map = df.groupby('Title')['Age'].median().to_dict()

df['Age'] = df['Age'].fillna(df['Title'].map(title_age_map))


## 5. Remove Duplicates
df = df.drop_duplicates()

## 6. Standardize Formats
# Example: string formatting
df['Sex'] = df['Sex'].str.strip().str.lower()
df['Embarked'] = df['Embarked'].str.strip().str.lower()

## 7. Outlier Detection & Treatment
# Example: IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
filtered = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) |(df['Fare'] > (Q3 + 1.5 * IQR)))]


## 8. Save Cleaned Dataset
filtered.to_csv("../data/cleaned_dataset.csv", index=False)
#after cleaning
#sample rows
print(df.info())
print(filtered.head())
print("Shape after Cleaning",filtered.shape)
df.describe()


Generated: 
Great! You've written a solid data-cleaning pipeline, but there are **several subtle bugs and logical issues** that could cause problems — especially around **missing value handling**, **inconsistent DataFrame usage**, and **data leakage**. Let’s go through them step by step.

---

### 🔴 **Critical Issue #1: You’re using `df` and `filtered` inconsistently**
- You clean `df` (handle missing values, duplicates, standardize).
- But for **outlier removal**, you create a **new DataFrame `filtered`**.
- Then you **save `filtered`**, but **print diagnostics on both `df` and `filtered`** → this is confusing and inconsistent.

✅ **Fix**: Either:
- **Option A (Recommended)**: Apply all steps to **one DataFrame** (`df`), overwriting it each time.
- **Option B**: Use `filtered` consistently after outlier step.

👉 **Why it matters**: Your "after cleaning" snapshot should reflect the **exact data you saved** — which is `filtered`. But you never standardized `filtered['Sex']` or `filtered['Embarked']`! Those steps were applied only to `df`.

---

### 🔴 **Critical Issue #2: `fillna(df['Embarked'].mode())` is wrong**
```python
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())
```
- `df['Embarked'].mode()` returns a **Series**, not a scalar.
- This will cause a **shape mismatch** or fill with `NaN`.

✅ **Fix**:
```python
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

---

### 🔴 **Issue #3: Outlier treatment drops rows, but you never update `df`**
You do:
```python
filtered = df[~((df['Fare'] < ...) | ...)]
```
But then:
- You **save `filtered`**
- But later call `df.describe()` → this describes the **pre-outlier-removal** data!

✅ **Fix**: Either:
- Reassign: `df = df[...]` (and drop `filtered`)
- Or use `filtered` everywhere after that point

---

### 🔴 **Issue #4: `df.head()` called twice unnecessarily**
Not a bug, but redundant. You can remove the first `df.head()` after loading.

---

### 🔴 **Issue #5: No check for missing values before imputation**
You assume `Age` and `Embarked` have missing values — which is fine for Titanic — but **good practice** is to check first (especially if dataset changes).

Not critical, but worth noting.

---

### 🔴 **Issue #6: You never handle missing `Fare` (if any)**
In Titanic, `Fare` has **1 missing value in the test set**, but usually **none in train**. However, if your `raw_dataset.csv` has missing `Fare`, the IQR step will fail or behave unexpectedly.

But since you didn’t impute `Fare`, and then use it in IQR, you might get errors if `NaN` exists.

✅ **Safer**: Impute `Fare` before outlier detection:
```python
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
```

---

### ✅ **Other Observations (Not Bugs, But Improvements)**
- You drop `Cabin` early — good.
- Title-based age imputation — excellent!
- Standardizing `Sex` and `Embarked` to lowercase — fine, but ensure your model expects that (sometimes `'S'` vs `'s'` matters if merging later).

---

### ✅ **Corrected & Cleaned Code**

```python
import pandas as pd
import numpy as np

## 2. Load Dataset
df = pd.read_csv("../data/raw_dataset.csv")

## 3. Initial Exploration
print("=== BEFORE CLEANING ===")
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSample rows:")
print(df.head())

## 4. Handle Missing Values
# Drop Cabin (too many missing)
df.drop(columns=['Cabin'], inplace=True)

# Fill Embarked with mode (most frequent)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fill Fare (in case of missing values)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Extract and standardize titles
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace([
    'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Impute Age based on Title
title_age_map = df.groupby('Title')['Age'].median().to_dict()
df['Age'] = df['Age'].fillna(df['Title'].map(title_age_map))

## 5. Remove Duplicates
df = df.drop_duplicates()

## 6. Standardize Formats
df['Sex'] = df['Sex'].str.strip().str.lower()
df['Embarked'] = df['Embarked'].str.strip().str.lower()

## 7. Outlier Detection & Treatment (on Fare)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

## 8. Save Cleaned Dataset
df.to_csv("../data/cleaned_dataset.csv", index=False)

## After Cleaning Snapshot
print("\n=== AFTER CLEANING ===")
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSample rows:")
print(df.head())
print("\nSummary stats:")
print(df.describe())
```

---

### ✅ Key Fixes Summary:
| Issue | Fix |
|------|-----|
| Inconsistent `df` vs `filtered` | Use only `df` throughout |
| `mode()` returns Series | Use `.mode()[0]` |
| Outlier step doesn’t update `df` | Reassign `df = df[...]` |
| No `Fare` imputation | Added median fill |
| Unclear before/after snapshots | Added structured print statements |

Now your pipeline is **consistent, correct, and reproducible**! 🎯

## Results
- Rows before: 
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S

- Rows after:
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Embarked,Title
1,0,3,"Braund, Mr. Owen Harris",male,22.0,1,0,A/5 21171,7.25,s,Mr
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38.0,1,0,PC 17599,71.2833,c,Mrs
3,1,3,"Heikkinen, Miss. Laina",female,26.0,0,0,STON/O2. 3101282,7.925,s,Miss
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35.0,1,0,113803,53.1,s,Mrs
5,0,3,"Allen, Mr. William Henry",male,35.0,0,0,373450,8.05,s,Mr

Video: https://youtu.be/gAUKM-505Ls
        