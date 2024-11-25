from .models.MSTpp import MST_Plus_Plus

all_models = [
    MST_Plus_Plus(model_path="models/MSTpp/weights.pt"),
]
all_tests = []

for model in all_models:
    for test in all_tests:
        test(model)
