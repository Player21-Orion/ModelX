def add_features(df):
    df["load_per_staff"] = df["restaurant_load"] / df["staff_available"]
    df["order_complexity"] = df["num_items"] * df["total_quantity"]
    return df
