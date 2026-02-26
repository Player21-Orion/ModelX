from src.data_generator import generate_data
from src.train import train_models
from src.predict import predict_sample

# Train model
df = generate_data(10000)
train_models(df)

# Test prediction
sample_input = {
    "num_items": 5,
    "total_quantity": 10,
    "order_value": 800,
    "restaurant_load": 30,
    "avg_prep_time": 20,
    "staff_available": 6,
    "is_weekend": 1,
    "is_rain": 0,
    "is_peak_hour": 1
}

prediction = predict_sample(sample_input)

print("\nPredicted Kitchen Prep Time:", round(prediction, 2), "minutes")
