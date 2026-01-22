from src.data.load import load_dataset, load_label
from src.data.clean import clean_text_df

def get_eda_dataframe(spark, raw_data_dir):
    # load raw
    bundle = load_dataset(spark, raw_data_dir)   # SimpleNamespace(df_true, df_fake)

    # label + combine
    df_all, _ = load_label(bundle.df_true, bundle.df_fake)

    # clean combined dataset
    df_all, _ = clean_text_df(df_all, cache=False)

    # cache for EDA reuse
    df_all = df_all.cache()
    _ = df_all.count()   # materialize cache

    return df_all
