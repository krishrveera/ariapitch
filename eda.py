import pandas as pd
import glob
import os
import warnings

warnings.filterwarnings('ignore')

def load_data(base_dir='adult'):
    phenotype_files = glob.glob(os.path.join(base_dir, 'phenotype', '**', '*.tsv'), recursive=True)
    feature_files = glob.glob(os.path.join(base_dir, 'features', '*.*'))
    
    dfs = {}
    
    print("Loading phenotype data...")
    for f in phenotype_files:
        name = os.path.basename(f).replace('.tsv', '')
        try:
            df = pd.read_csv(f, sep='\t', low_memory=False)
            
            # Standardize session_id column names if possible to improve join rate
            for col in df.columns:
                if col.endswith('session_id') and col != 'session_id':
                    df = df.rename(columns={col: 'session_id'})
                    
            dfs[name] = df
            print(f" Loaded {name} ({df.shape[0]} rows, {df.shape[1]} cols)")
        except Exception as e:
            print(f" Failed to load {f}: {e}")
            
    print("\nLoading feature data...")
    for f in feature_files:
        if f.endswith('.tsv'):
            name = os.path.basename(f).replace('.tsv', '')
            try:
                df = pd.read_csv(f, sep='\t', low_memory=False)
                dfs[name] = df
                print(f" Loaded {name} ({df.shape[0]} rows, {df.shape[1]} cols)")
            except Exception as e:
                print(f" Failed to load {f}: {e}")
        elif f.endswith('.parquet'):
            name = os.path.basename(f).replace('.parquet', '')
            # Skip massive files to prevent OOM during join unless specifically requested.
            size_mb = os.path.getsize(f) / (1024 * 1024)
            if size_mb > 500:
                print(f" Skipping {name} due to large size ({size_mb:.1f} MB)")
                continue
                
            try:
                df = pd.read_parquet(f)
                dfs[name] = df
                print(f" Loaded {name} ({df.shape[0]} rows, {df.shape[1]} cols)")
            except Exception as e:
                print(f" Failed to load {f}: {e}")

    return dfs

def merge_all_data(dfs):
    print("\nMerging all dataframes...")
    if not dfs:
        return pd.DataFrame()
        
    # Sort dataframes by size (descending) so we use the largest (recording level) as the base table
    df_list = sorted(list(dfs.values()), key=lambda x: len(x), reverse=True)
    
    final_df = df_list[0]
    
    for i, df in enumerate(df_list[1:], 1):
        common_cols = list(set(final_df.columns).intersection(set(df.columns)))
        # Prefer merging on participant_id, session_id, and task_name
        merge_keys = [c for c in ['participant_id', 'session_id', 'task_name'] if c in common_cols]
        
        if not merge_keys:
            if 'participant_id' in final_df.columns and 'participant_id' in df.columns:
                merge_keys = ['participant_id']
            else:
                print(f" Warning: No common merge keys found for a dataframe, skipping.")
                continue
                
        # Drop intersecting columns that are not join keys to avoid duplicate column suffixes (_x, _y)
        cols_to_drop = [c for c in common_cols if c not in merge_keys]
        df = df.drop(columns=cols_to_drop)
        
        # CRITICAL FIX: Drop duplicates on the merge keys in the right table to avoid cartesian explosions!
        # Many tables have 1 row per participant, but if they have a rare duplicate, it exponentially multiplies rows.
        df = df.drop_duplicates(subset=merge_keys)
        
        for key in merge_keys:
            final_df[key] = final_df[key].astype(str)
            df[key] = df[key].astype(str)
        
        try:
            final_df = pd.merge(final_df, df, on=merge_keys, how='left')
        except Exception as e:
            print(f" Failed to merge a dataframe: {e}")
            
    return final_df

def main():
    dfs = load_data('adult')
    if not dfs:
        print("No data loaded.")
        return
        
    final_df = merge_all_data(dfs)
    print(f"\nFinal merged dataframe shape: {final_df.shape}")
    print("\nColumns (first 50):")
    print(final_df.columns.tolist()[:50])
    
    # Save the fully merged DF to a fast storage format or CSV if needed
    # final_df.to_csv('merged_adult_data.csv', index=False)
    # print("Saved merged data to merged_adult_data.csv")
    
    # Now the researcher can perform EDA on final_df here.
    return final_df

if __name__ == '__main__':
    main()
