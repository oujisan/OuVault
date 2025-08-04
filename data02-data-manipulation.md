# [ai] #02 - ML Data Manipulation with NumPy and Pandas

![data](https://raw.githubusercontent.com/oujisan/OuVault/main/img/data.png)

## Data Manipulation with NumPy and Pandas

---

Setelah memahami dasar-dasar data, sekarang saatnya menguasai tools yang akan menjadi senjata utama kita: NumPy dan Pandas. Kedua library ini adalah backbone dari data science dan machine learning di Python.

## NumPy: The Foundation of Numerical Computing

---

NumPy (Numerical Python) adalah library fundamental untuk scientific computing. Hampir semua library machine learning bergantung pada NumPy.

### Why NumPy?

- **Performance**: 10-100x lebih cepat dari Python lists
- **Memory efficient**: Menggunakan memori lebih sedikit
- **Vectorization**: Operasi pada seluruh array tanpa loop explicit
- **Broadcasting**: Operasi pada arrays dengan shape berbeda

### Creating NumPy Arrays

```python
import numpy as np

# Dari Python list
arr_from_list = np.array([1, 2, 3, 4, 5])
print("From list:", arr_from_list)

# Array 2D
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr_2d)

# Built-in functions
zeros = np.zeros((3, 4))  # Array berisi nol
ones = np.ones((2, 3))    # Array berisi satu
arange = np.arange(0, 10, 2)  # Range dengan step
linspace = np.linspace(0, 1, 5)  # 5 titik dari 0 ke 1

print("Zeros:\n", zeros)
print("Arange:", arange)
print("Linspace:", linspace)
```

### Array Attributes and Properties

```python
# Sample array untuk eksplorasi
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Array:\n", data)
print("Shape:", data.shape)        # Dimensi array
print("Size:", data.size)          # Total elemen
print("Ndim:", data.ndim)          # Jumlah dimensi
print("Dtype:", data.dtype)        # Tipe data
print("Itemsize:", data.itemsize)  # Ukuran per elemen (bytes)
```

### Array Indexing and Slicing

```python
# 1D Array indexing
arr_1d = np.array([10, 20, 30, 40, 50])

print("Element pertama:", arr_1d[0])
print("Element terakhir:", arr_1d[-1])
print("Slice 1-3:", arr_1d[1:4])
print("Every second element:", arr_1d[::2])

# 2D Array indexing
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("\n2D Array:")
print(arr_2d)
print("Element [1,2]:", arr_2d[1, 2])  # Baris 1, kolom 2
print("Baris pertama:", arr_2d[0, :])   # Semua kolom dari baris 0
print("Kolom terakhir:", arr_2d[:, -1]) # Semua baris dari kolom terakhir
```

### Boolean Indexing (Advanced Selection)

```python
# Sample data
scores = np.array([85, 92, 78, 96, 73, 88, 91])
names = np.array(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace'])

# Boolean mask
high_scores = scores > 85
print("High scores mask:", high_scores)
print("Students with high scores:", names[high_scores])
print("Their scores:", scores[high_scores])

# Complex conditions
excellent_scores = (scores > 90) & (scores < 95)
print("Excellent students:", names[excellent_scores])
```

### Array Operations and Broadcasting

```python
# Basic operations
arr = np.array([1, 2, 3, 4, 5])

print("Original:", arr)
print("Add 10:", arr + 10)         # Broadcasting
print("Multiply by 2:", arr * 2)
print("Square:", arr ** 2)

# Array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print("Addition:", arr1 + arr2)
print("Dot product:", np.dot(arr1, arr2))

# Broadcasting dengan different shapes
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])

print("Matrix:\n", matrix)
print("Vector:", vector)
print("Matrix + Vector:\n", matrix + vector)  # Broadcasting magic!
```

### Statistical Operations

```python
# Sample dataset
data = np.array([[85, 92, 78], [96, 73, 88], [91, 86, 94]])
print("Grades dataset:\n", data)

# Basic statistics
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard deviation:", np.std(data))
print("Min:", np.min(data))
print("Max:", np.max(data))

# Axis-wise operations
print("Mean per student (axis=1):", np.mean(data, axis=1))
print("Mean per subject (axis=0):", np.mean(data, axis=0))
```

## Pandas: Data Analysis Powerhouse

---

Pandas built on top of NumPy dan menyediakan struktur data yang lebih user-friendly untuk analisis data.

### Core Data Structures

#### Series: 1D labeled data

```python
import pandas as pd

# Creating Series
grades = pd.Series([85, 92, 78, 96, 73], 
                  index=['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
                  name='Math_Scores')

print("Grades Series:")
print(grades)
print("\nAlice's grade:", grades['Alice'])
print("Students with grade > 80:", grades[grades > 80])
```

#### DataFrame: 2D labeled data

```python
# Creating DataFrame from dictionary
student_data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [20, 22, 21, 23, 20],
    'Math': [85, 92, 78, 96, 73],
    'Science': [88, 85, 92, 89, 94],
    'English': [92, 88, 85, 93, 89]
}

df = pd.DataFrame(student_data)
print("Student DataFrame:")
print(df)
```

### DataFrame Exploration and Information

```python
# Basic information
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Index:", df.index.tolist())

# Quick overview
print("\nDataFrame Info:")
print(df.info())

print("\nFirst 3 rows:")
print(df.head(3))

print("\nLast 2 rows:")
print(df.tail(2))

print("\nDescriptive statistics:")
print(df.describe())
```

### Data Selection and Filtering

```python
# Column selection
print("Names only:")
print(df['Name'])

print("\nMultiple columns:")
print(df[['Name', 'Math']])

# Row selection by index
print("\nFirst student:")
print(df.iloc[0])  # By position

print("\nStudent by label:")
print(df.loc[0])   # By label (same as iloc here)

# Boolean filtering
high_math = df[df['Math'] > 85]
print("\nStudents with Math > 85:")
print(high_math)

# Complex filtering
good_students = df[(df['Math'] > 80) & (df['Science'] > 85)]
print("\nGood students (Math>80 AND Science>85):")
print(good_students)
```

### Data Manipulation and Transformation

```python
# Adding new columns
df['Total'] = df['Math'] + df['Science'] + df['English']
df['Average'] = df['Total'] / 3
df['Grade'] = df['Average'].apply(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C')

print("DataFrame with new columns:")
print(df)

# Sorting
df_sorted = df.sort_values('Average', ascending=False)
print("\nSorted by Average:")
print(df_sorted)

# Grouping and aggregation
grade_summary = df.groupby('Grade').agg({
    'Math': ['mean', 'min', 'max'],
    'Science': ['mean', 'min', 'max'],
    'Average': 'count'
})
print("\nGrade summary:")
print(grade_summary)
```

### Handling Missing Data

```python
# Create data with missing values
messy_data = {
    'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
    'Age': [20, None, 21, 23, 20],
    'Salary': [50000, 60000, None, 70000, 55000],
    'Department': ['IT', 'Finance', 'IT', None, 'HR']
}

df_messy = pd.DataFrame(messy_data)
print("Messy data:")
print(df_messy)

# Check missing values
print("\nMissing values:")
print(df_messy.isnull().sum())

print("\nMissing values percentage:")
print((df_messy.isnull().sum() / len(df_messy)) * 100)

# Handling missing values
# 1. Drop rows with any missing value
df_dropna = df_messy.dropna()
print("\nAfter dropping rows with NaN:")
print(df_dropna)

# 2. Fill missing values
df_filled = df_messy.copy()
df_filled['Age'].fillna(df_filled['Age'].mean(), inplace=True)  # Fill with mean
df_filled['Name'].fillna('Unknown', inplace=True)              # Fill with string
df_filled['Department'].fillna('General', inplace=True)        # Fill with default

print("\nAfter filling missing values:")
print(df_filled)
```

### String Operations

```python
# Sample text data
text_data = pd.DataFrame({
    'names': ['john doe', 'JANE SMITH', 'Bob Johnson', 'alice brown'],
    'emails': ['john@email.com', 'JANE@COMPANY.COM', 'bob@test.org', 'alice@uni.edu']
})

print("Original text data:")
print(text_data)

# String operations
text_data['names_clean'] = text_data['names'].str.title()  # Title case
text_data['email_domain'] = text_data['emails'].str.split('@').str[1]  # Extract domain
text_data['email_lower'] = text_data['emails'].str.lower()  # Lowercase

print("\nAfter string operations:")
print(text_data)

# String filtering
gmail_users = text_data[text_data['emails'].str.contains('gmail', case=False)]
print("\nGmail users:")
print(gmail_users)
```

### Date and Time Operations

```python
# Sample datetime data
dates = pd.date_range('2024-01-01', periods=6, freq='D')
sales_data = pd.DataFrame({
    'date': dates,
    'sales': [100, 150, 120, 200, 180, 160],
    'customers': [10, 15, 12, 20, 18, 16]
})

print("Sales data:")
print(sales_data)

# Datetime operations
sales_data['day_name'] = sales_data['date'].dt.day_name()
sales_data['month'] = sales_data['date'].dt.month
sales_data['is_weekend'] = sales_data['date'].dt.dayofweek >= 5

print("\nWith datetime features:")
print(sales_data)

# Time-based filtering
january_data = sales_data[sales_data['date'].dt.month == 1]
print("\nJanuary data:")
print(january_data)
```

## Advanced Pandas Operations

---

### Merging and Joining DataFrames

```python
# Sample DataFrames
students = pd.DataFrame({
    'student_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'major': ['Math', 'CS', 'Physics', 'Chemistry']
})

grades = pd.DataFrame({
    'student_id': [1, 2, 3, 5],
    'course': ['Calculus', 'Programming', 'Mechanics', 'Organic'],
    'grade': [85, 92, 78, 88]
})

print("Students:")
print(students)
print("\nGrades:")
print(grades)

# Inner join (default)
inner_join = pd.merge(students, grades, on='student_id')
print("\nInner join:")
print(inner_join)

# Left join
left_join = pd.merge(students, grades, on='student_id', how='left')
print("\nLeft join:")
print(left_join)

# Outer join
outer_join = pd.merge(students, grades, on='student_id', how='outer')
print("\nOuter join:")
print(outer_join)
```

### Pivot Tables and Cross-tabulation

```python
# Sample sales data
sales = pd.DataFrame({
    'region': ['North', 'South', 'East', 'West', 'North', 'South'],
    'product': ['A', 'A', 'B', 'B', 'B', 'A'],
    'quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2'],
    'sales': [100, 150, 200, 120, 180, 160]
})

print("Sales data:")
print(sales)

# Pivot table
pivot = sales.pivot_table(values='sales', 
                         index='region', 
                         columns='quarter', 
                         aggfunc='sum', 
                         fill_value=0)
print("\nPivot table:")
print(pivot)

# Cross-tabulation
crosstab = pd.crosstab(sales['region'], sales['product'], values=sales['sales'], aggfunc='sum')
print("\nCross-tabulation:")
print(crosstab)
```

### Data Type Optimization

```python
# Sample data with inefficient types
big_data = pd.DataFrame({
    'id': range(1000),
    'category': ['A', 'B', 'C'] * 334,  # Repeated categories
    'value': np.random.randn(1000),
    'flag': [True, False] * 500
})

print("Original memory usage:")
print(big_data.memory_usage(deep=True))

# Optimize data types
big_data['id'] = big_data['id'].astype('int16')  # Smaller integer
big_data['category'] = big_data['category'].astype('category')  # Categorical
big_data['flag'] = big_data['flag'].astype('bool')  # Boolean

print("Optimized memory usage:")
print(big_data.memory_usage(deep=True))

print("Data types:")
print(big_data.dtypes)
```

## Performance Tips and Best Practices

---

### Vectorization vs Loops

```python
import time

# Sample large dataset
large_data = pd.DataFrame({
    'values': np.random.randn(100000)
})

# BAD: Using loops
start_time = time.time()
result_loop = []
for value in large_data['values']:
    result_loop.append(value ** 2 if value > 0 else 0)
loop_time = time.time() - start_time

# GOOD: Using vectorization
start_time = time.time()
result_vectorized = np.where(large_data['values'] > 0, 
                           large_data['values'] ** 2, 
                           0)
vectorized_time = time.time() - start_time

print(f"Loop method: {loop_time:.4f} seconds")
print(f"Vectorized method: {vectorized_time:.4f} seconds")
print(f"Speedup: {loop_time/vectorized_time:.1f}x faster")
```

### Efficient Data Loading

```python
# Tips for reading large files
def read_large_csv_efficiently(filename):
    """
    Tips untuk membaca CSV besar secara efisien
    """
    # 1. Specify data types
    dtypes = {
        'id': 'int32',
        'category': 'category',
        'value': 'float32'
    }
    
    # 2. Read in chunks for very large files
    chunk_size = 10000
    chunks = []
    
    # Simulasi membaca chunks
    for i in range(3):  # Simulate 3 chunks
        chunk_data = {
            'id': range(i*chunk_size, (i+1)*chunk_size),
            'category': ['A', 'B', 'C'] * (chunk_size//3),
            'value': np.random.randn(chunk_size)
        }
        chunk_df = pd.DataFrame(chunk_data)
        chunks.append(chunk_df)
    
    # Combine chunks
    df = pd.concat(chunks, ignore_index=True)
    return df

# Example usage
efficient_df = read_large_csv_efficiently('large_file.csv')
print(f"Loaded {len(efficient_df)} rows efficiently")
```

### Memory Management

```python
# Monitor memory usage
def check_memory_usage(df, name="DataFrame"):
    """Check memory usage of DataFrame"""
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"{name} memory usage: {memory_mb:.2f} MB")
    return memory_mb

# Sample operations with memory monitoring
df_sample = pd.DataFrame({
    'text': ['hello world'] * 10000,
    'numbers': range(10000),
    'floats': np.random.randn(10000)
})

check_memory_usage(df_sample, "Original")

# Optimize
df_sample['text'] = df_sample['text'].astype('category')
df_sample['numbers'] = df_sample['numbers'].astype('int16')
df_sample['floats'] = df_sample['floats'].astype('float32')

check_memory_usage(df_sample, "Optimized")
```

## Data Validation and Quality Checks

---

### Comprehensive Data Validation

```python
def comprehensive_data_check(df, name="Dataset"):
    """Comprehensive data quality check"""
    print(f"🔍 Data Quality Report: {name}")
    print("=" * 50)
    
    # Basic info
    print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n❌ Missing Values:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"   • {col}: {count:,} ({pct:.1f}%)")
    else:
        print(f"\n✅ No missing values found")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\n⚠️ Duplicate rows: {duplicates:,}")
    else:
        print(f"\n✅ No duplicate rows")
    
    # Data types
    print(f"\n📋 Data Types:")
    for col, dtype in df.dtypes.items():
        unique_count = df[col].nunique()
        print(f"   • {col}: {dtype} ({unique_count:,} unique values)")
    
    # Outliers detection (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Potential Outliers (using IQR method):")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                pct = (len(outliers) / len(df)) * 100
                print(f"   • {col}: {len(outliers):,} outliers ({pct:.1f}%)")

# Test the validation function
test_data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 1],  # Duplicate
    'name': ['Alice', 'Bob', None, 'Diana', 'Eve', 'Alice'],  # Missing value
    'age': [25, 30, 28, 150, 22, 25],  # Outlier (150)
    'salary': [50000, 60000, 55000, 70000, 45000, 50000]
})

comprehensive_data_check(test_data, "Sample Employee Data")
```

## Real-world Data Cleaning Pipeline

---

### Complete Data Cleaning Workflow

```python
def clean_dataset(df, target_column=None):
    """
    Complete data cleaning pipeline
    """
    print("🧹 Starting data cleaning pipeline...")
    original_shape = df.shape
    
    # 1. Remove completely empty rows/columns
    df = df.dropna(how='all')  # Remove rows that are completely empty
    df = df.dropna(axis=1, how='all')  # Remove columns that are completely empty
    
    print(f"✅ Removed empty rows/columns: {original_shape} → {df.shape}")
    
    # 2. Handle duplicates
    before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before_dedup - len(df)
    if duplicates_removed > 0:
        print(f"✅ Removed {duplicates_removed} duplicate rows")
    
    # 3. Data type optimization
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if it's actually numeric
            try:
                numeric_version = pd.to_numeric(df[col], errors='coerce')
                if not numeric_version.isnull().all():
                    df[col] = numeric_version
                    print(f"✅ Converted {col} to numeric")
            except:
                pass
            
            # Check if it should be categorical
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
                print(f"✅ Converted {col} to category")
    
    # 4. Handle missing values intelligently
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_pct > 50:
                print(f"⚠️ {col} has {missing_pct:.1f}% missing values - consider dropping")
            else:
                if df[col].dtype in ['int64', 'float64']:
                    # Fill numeric with median
                    df[col].fillna(df[col].median(), inplace=True)
                    print(f"✅ Filled {col} missing values with median")
                elif df[col].dtype == 'category' or df[col].dtype == 'object':
                    # Fill categorical with mode
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
                    print(f"✅ Filled {col} missing values with mode")
    
    print(f"🎉 Data cleaning completed! Final shape: {df.shape}")
    return df

# Test the cleaning pipeline
dirty_data = pd.DataFrame({
    'id': [1, 2, 3, None, 5, 1, 7],
    'name': ['Alice', 'Bob', None, 'Diana', 'Eve', 'Alice', 'Frank'],
    'age': ['25', '30', '28', '35', 'invalid', '25', '40'],
    'department': ['IT', 'Finance', 'IT', 'HR', 'Finance', 'IT', None],
    'salary': [50000, None, 55000, 70000, 45000, 50000, 60000],
    'empty_col': [None, None, None, None, None, None, None]
})

print("Original dirty data:")
print(dirty_data)
print(dirty_data.dtypes)

cleaned_data = clean_dataset(dirty_data)
print("\nCleaned data:")
print(cleaned_data)
print(cleaned_data.dtypes)
```

## Advanced Indexing and Performance

---

### Multi-level Indexing (MultiIndex)

```python
# Creating hierarchical data
sales_data = pd.DataFrame({
    'Year': [2023, 2023, 2023, 2024, 2024, 2024],
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q1', 'Q2', 'Q3'],
    'Region': ['North', 'North', 'South', 'North', 'South', 'South'],
    'Sales': [100, 150, 200, 120, 180, 160],
    'Customers': [10, 15, 20, 12, 18, 16]
})

# Set multi-level index
multi_df = sales_data.set_index(['Year', 'Quarter', 'Region'])
print("Multi-level indexed data:")
print(multi_df)

# Accessing multi-level data
print("\n2023 data:")
print(multi_df.loc[2023])

print("\n2024 Q2 data:")
print(multi_df.loc[(2024, 'Q2')])

# Cross-section
print("\nAll Q1 data across years:")
print(multi_df.xs('Q1', level='Quarter'))
```

### Efficient Groupby Operations

```python
# Large dataset simulation for performance testing
np.random.seed(42)
large_sales = pd.DataFrame({
    'product': np.random.choice(['A', 'B', 'C', 'D'], 10000),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 10000),
    'sales': np.random.randn(10000) * 1000 + 5000,
    'date': pd.date_range('2023-01-01', periods=10000, freq='H')
})

# Efficient groupby operations
print("Groupby aggregations:")

# Multiple aggregations at once
agg_result = large_sales.groupby('product').agg({
    'sales': ['mean', 'sum', 'count', 'std'],
    'region': lambda x: x.mode().iloc[0]  # Most common region
})

print(agg_result.head())

# Time-based grouping
large_sales['month'] = large_sales['date'].dt.to_period('M')
monthly_sales = large_sales.groupby(['month', 'product'])['sales'].sum().unstack()

print("\nMonthly sales by product:")
print(monthly_sales.head())
```

## Summary dan Best Practices

---

### 🎯 Key Takeaways dari NumPy:

1. **Arrays** adalah struktur data fundamental untuk numerical computing
2. **Broadcasting** memungkinkan operasi efisien pada arrays dengan shape berbeda
3. **Vectorization** jauh lebih cepat daripada Python loops
4. **Boolean indexing** sangat powerful untuk filtering data

### 🎯 Key Takeaways dari Pandas:

1. **DataFrame** adalah struktur utama untuk structured data
2. **Series** perfect untuk 1D data dengan labels
3. **Groupby operations** essential untuk data aggregation
4. **Missing data handling** crucial untuk data quality
5. **Data type optimization** dapat menghemat memory significantly

### 🔧 Best Practices:

```python
# DO's and DON'Ts Summary
print("✅ DO's:")
print("• Use vectorized operations instead of loops")
print("• Optimize data types for memory efficiency")
print("• Handle missing values appropriately")
print("• Use categorical data type for repeated strings")
print("• Validate data quality regularly")

print("\n❌ DON'Ts:")
print("• Don't use loops when vectorization is possible")
print("• Don't ignore missing values")
print("• Don't load entire large files into memory at once")
print("• Don't forget to check for duplicates")
print("• Don't skip data type optimization")
```

### 🔜 Next Steps:

Sekarang setelah menguasai manipulasi data dengan NumPy dan Pandas, kita siap untuk mempelajari statistik deskriptif yang akan membantu kita memahami karakteristik data kita secara mendalam. Ini adalah langkah penting sebelum membangun model machine learning.