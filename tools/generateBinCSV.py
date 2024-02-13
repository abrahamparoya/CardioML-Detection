import pandas as pd
def main():
    
    exams = pd.read_csv("../balanced_5000/balanced_5000.csv")
   # Copy the DataFrame to avoid SettingWithCopyWarning

    boolean_column_names = ['1dAVb','RBBB','LBBB','SB','ST','AF']

    # Create a new DataFrame with only the specified boolean columns
    bool_columns_only = exams[boolean_column_names].copy()

    # Replace True/False values with 1/0 in the boolean columns
    bool_columns_only = bool_columns_only.astype(int)

    # Save the modified DataFrame to a new CSV file
    bool_columns_only.to_csv("../balanced_5000/balanced_5000_bin.csv", index=False)
    
if __name__ == "__main__":
    main()
