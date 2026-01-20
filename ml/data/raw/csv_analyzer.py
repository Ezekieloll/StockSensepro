import pandas as pd
import numpy as np
from datetime import datetime

def analyze_csv(file_path):
    """
    Comprehensive CSV analysis for forecasting projects
    Usage: analyze_csv('your_data.csv')
    """
    
    print("="*70)
    print("📊 CSV DATA ANALYSIS REPORT")
    print("="*70)
    
    # Load the CSV
    try:
        df = pd.read_csv(file_path)
        print(f"\n✅ File loaded successfully: {file_path}\n")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # ==========================================
    # 1. BASIC INFORMATION
    # ==========================================
    print("\n" + "="*70)
    print("1️⃣  BASIC INFORMATION")
    print("="*70)
    print(f"Total Rows:        {len(df):,}")
    print(f"Total Columns:     {len(df.columns)}")
    print(f"Memory Usage:      {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicate Rows:    {df.duplicated().sum():,}")
    
    # ==========================================
    # 2. COLUMN DETAILS
    # ==========================================
    print("\n" + "="*70)
    print("2️⃣  COLUMN DETAILS")
    print("="*70)
    
    column_info = []
    for col in df.columns:
        info = {
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null': f"{df[col].count():,}",
            'Null': f"{df[col].isna().sum():,}",
            'Null %': f"{(df[col].isna().sum() / len(df) * 100):.1f}%",
            'Unique': f"{df[col].nunique():,}"
        }
        column_info.append(info)
    
    col_df = pd.DataFrame(column_info)
    print(col_df.to_string(index=False))
    
    # ==========================================
    # 3. DATA TYPES SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("3️⃣  DATA TYPES SUMMARY")
    print("="*70)
    print(df.dtypes.value_counts())
    
    # ==========================================
    # 4. DETECT DATE COLUMNS
    # ==========================================
    print("\n" + "="*70)
    print("4️⃣  DATE/TIME COLUMNS DETECTED")
    print("="*70)
    
    date_columns = []
    for col in df.columns:
        # Try to parse as datetime
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(100), errors='coerce')
                # If most values can be parsed, it's likely a date
                sample = pd.to_datetime(df[col].head(1000), errors='coerce')
                if sample.notna().sum() / len(sample) > 0.8:
                    date_columns.append(col)
                    
                    # Get date range
                    dates = pd.to_datetime(df[col], errors='coerce')
                    min_date = dates.min()
                    max_date = dates.max()
                    date_range = (max_date - min_date).days
                    
                    print(f"\n📅 {col}:")
                    print(f"   Min Date:    {min_date}")
                    print(f"   Max Date:    {max_date}")
                    print(f"   Date Range:  {date_range} days ({date_range/365:.1f} years)")
                    print(f"   Frequency:   ~{len(df) / max(date_range, 1):.1f} records per day")
            except:
                pass
    
    if not date_columns:
        print("⚠️  No date columns detected")
        print("   (Date columns are essential for time-series forecasting)")
    
    # ==========================================
    # 5. NUMERICAL COLUMNS ANALYSIS
    # ==========================================
    print("\n" + "="*70)
    print("5️⃣  NUMERICAL COLUMNS STATISTICS")
    print("="*70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        print(df[numeric_cols].describe().round(2).to_string())
        
        # Check for negative values (important for quantity/sales)
        print("\n📊 Additional Checks:")
        for col in numeric_cols:
            negatives = (df[col] < 0).sum()
            zeros = (df[col] == 0).sum()
            print(f"   {col}:")
            print(f"      Negative values: {negatives:,} ({negatives/len(df)*100:.1f}%)")
            print(f"      Zero values:     {zeros:,} ({zeros/len(df)*100:.1f}%)")
    else:
        print("⚠️  No numerical columns found")
    
    # ==========================================
    # 6. CATEGORICAL COLUMNS ANALYSIS
    # ==========================================
    print("\n" + "="*70)
    print("6️⃣  CATEGORICAL COLUMNS (Top Values)")
    print("="*70)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Remove date columns from categorical
    categorical_cols = [col for col in categorical_cols if col not in date_columns]
    
    if categorical_cols:
        for col in categorical_cols[:5]:  # Show first 5 categorical columns
            print(f"\n📁 {col} (Unique: {df[col].nunique()}):")
            top_values = df[col].value_counts().head(5)
            for val, count in top_values.items():
                print(f"   {val}: {count:,} ({count/len(df)*100:.1f}%)")
    else:
        print("⚠️  No categorical columns found")
    
    # ==========================================
    # 7. SAMPLE DATA
    # ==========================================
    print("\n" + "="*70)
    print("7️⃣  SAMPLE DATA (First 5 Rows)")
    print("="*70)
    print(df.head().to_string())
    
    # ==========================================
    # 8. POTENTIAL ID COLUMNS
    # ==========================================
    print("\n" + "="*70)
    print("8️⃣  POTENTIAL IDENTIFIER COLUMNS")
    print("="*70)
    
    id_keywords = ['id', 'key', 'code', 'number', 'store', 'product', 'sku', 'item']
    potential_ids = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in id_keywords):
            potential_ids.append(col)
            print(f"   • {col}: {df[col].nunique():,} unique values")
    
    if not potential_ids:
        print("⚠️  No obvious ID columns detected")
    
    # ==========================================
    # 9. DATA QUALITY ISSUES
    # ==========================================
    print("\n" + "="*70)
    print("9️⃣  DATA QUALITY CHECKS")
    print("="*70)
    
    issues = []
    
    # Check for high null percentage
    high_null_cols = df.columns[df.isna().sum() / len(df) > 0.5].tolist()
    if high_null_cols:
        issues.append(f"⚠️  Columns with >50% missing: {', '.join(high_null_cols)}")
    
    # Check for single value columns (useless)
    single_val_cols = [col for col in df.columns if df[col].nunique() == 1]
    if single_val_cols:
        issues.append(f"⚠️  Columns with only 1 unique value: {', '.join(single_val_cols)}")
    
    # Check for mostly unique columns (might be IDs misclassified)
    mostly_unique = [col for col in df.columns if df[col].nunique() / len(df) > 0.95]
    if mostly_unique:
        issues.append(f"ℹ️  Mostly unique columns (possible IDs): {', '.join(mostly_unique)}")
    
    if not issues:
        print("✅ No major data quality issues detected")
    else:
        for issue in issues:
            print(f"   {issue}")
    
    # ==========================================
    # 10. FORECASTING READINESS
    # ==========================================
    print("\n" + "="*70)
    print("🔮 FORECASTING READINESS ASSESSMENT")
    print("="*70)
    
    readiness_score = 0
    total_checks = 5
    
    # Check 1: Has date column
    if date_columns:
        print("✅ Has date/time column")
        readiness_score += 1
    else:
        print("❌ Missing date/time column (REQUIRED for forecasting)")
    
    # Check 2: Has numerical target variable
    if numeric_cols:
        print(f"✅ Has numerical columns: {', '.join(numeric_cols[:3])}")
        readiness_score += 1
    else:
        print("❌ No numerical columns (need quantity/sales/demand)")
    
    # Check 3: Has categorical grouping (store, product, etc.)
    if potential_ids:
        print(f"✅ Has grouping variables: {', '.join(potential_ids[:3])}")
        readiness_score += 1
    else:
        print("⚠️  No clear grouping variables")
    
    # Check 4: Sufficient data points
    if len(df) > 100:
        print(f"✅ Sufficient data: {len(df):,} rows")
        readiness_score += 1
    else:
        print(f"⚠️  Limited data: {len(df):,} rows (need >100 for good forecasting)")
    
    # Check 5: Low missing data
    overall_missing = df.isna().sum().sum() / (len(df) * len(df.columns))
    if overall_missing < 0.1:
        print(f"✅ Low missing data: {overall_missing*100:.1f}%")
        readiness_score += 1
    else:
        print(f"⚠️  High missing data: {overall_missing*100:.1f}%")
    
    print(f"\n📊 Readiness Score: {readiness_score}/{total_checks}")
    
    if readiness_score >= 4:
        print("🎉 Dataset is READY for forecasting!")
    elif readiness_score >= 3:
        print("✓  Dataset is MOSTLY ready (minor improvements needed)")
    else:
        print("⚠️  Dataset needs SIGNIFICANT preparation before forecasting")
    
    # ==========================================
    # 11. RECOMMENDED NEXT STEPS
    # ==========================================
    print("\n" + "="*70)
    print("💡 RECOMMENDED NEXT STEPS")
    print("="*70)
    
    if not date_columns:
        print("1. ❗ ADD or IDENTIFY a date column (essential)")
    if df.isna().sum().sum() > 0:
        print("2. 🔧 Handle missing values (imputation or removal)")
    if df.duplicated().sum() > 0:
        print("3. 🔧 Remove or investigate duplicate rows")
    if not potential_ids:
        print("4. ℹ️  Identify grouping variables (store_id, product_id, etc.)")
    
    print("5. 📊 Aggregate data to appropriate time frequency (daily/weekly/monthly)")
    print("6. 🎯 Select target variable for forecasting")
    print("7. 🔍 Perform exploratory data analysis (EDA)")
    print("8. 🤖 Choose appropriate forecasting model")
    
    print("\n" + "="*70)
    print("📋 ANALYSIS COMPLETE")
    print("="*70)
    
    return df


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Replace with your actual CSV file path
    file_path = "transactions_3stores_2023_fullyear.csv"
    
    # Run analysis
    df = analyze_csv(file_path)
    
    # Optional: Save report to text file
    # import sys
    # sys.stdout = open('csv_analysis_report.txt', 'w')
    # df = analyze_csv(file_path)
    # sys.stdout.close()