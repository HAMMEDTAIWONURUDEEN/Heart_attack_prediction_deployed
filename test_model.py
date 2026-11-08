import joblib

# Load the trained model and scaler
model = joblib.load("heart_attack_model.pkl")
scaler = joblib.load("heart_attack_scaler.pkl")

# Check their properties
print("Model loaded successfully!")
print("Scaler loaded successfully!")

print("Model features:", model.n_features_in_)
print("Scaler features:", scaler.n_features_in_)

print("Scaler fitted:", hasattr(scaler, "mean_"))
print("Model classes:", model.classes_)

if hasattr(scaler, "feature_names_in_"):
    print("Scaler feature order:", scaler.feature_names_in_)