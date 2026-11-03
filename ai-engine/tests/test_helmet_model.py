from pathlib import Path
import onnxruntime as ort

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "helmet"
    / "best.onnx"
)

print("Model exists:", MODEL_PATH.exists())

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

print("\nMODEL LOADED SUCCESSFULLY!")

print("\nInput details:")
for item in session.get_inputs():
    print("Name:", item.name)
    print("Shape:", item.shape)
    print("Type:", item.type)

print("\nOutput details:")
for item in session.get_outputs():
    print("Name:", item.name)
    print("Shape:", item.shape)
    print("Type:", item.type)

print("\nAvailable providers:")
print(session.get_providers())