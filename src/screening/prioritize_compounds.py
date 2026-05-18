
def rank_by_predicted_activity(df):
    return df.sort_values('predicted_pIC50', ascending=False)
