import pandas as pd
import numpy as np

def generate_data(n=10000):
    np.random.seed(42)

    data = pd.DataFrame({
        "num_items": np.random.randint(1, 12, n),
        "total_quantity": np.random.randint(1, 25, n),
        "order_value": np.random.randint(100, 2000, n),
        "restaurant_load": np.random.randint(1, 60, n),
        "avg_prep_time": np.random.randint(8, 45, n),
        "staff_available": np.random.randint(2, 20, n),
        "is_weekend": np.random.randint(0, 2, n),
        "is_rain": np.random.randint(0, 2, n),
        "is_peak_hour": np.random.randint(0, 2, n)
    })

    # More realistic target creation
    data["kpt"] = (
        0.6 * data["num_items"] +
        0.25 * data["restaurant_load"] +
        0.3 * data["avg_prep_time"] -
        0.5 * data["staff_available"] +
        6 * data["is_peak_hour"] +
        4 * data["is_weekend"] +
        np.random.normal(0, 4, n)
    )

    return data