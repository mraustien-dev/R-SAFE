from pathlib import Path
import onnxruntime as ort

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "helmet"
    / "best.onnx"
)

print("=" * 60)
print("R-SAFE ONNX MODEL INSPECTION")
print("=" * 60)
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find model: {MODEL_PATH}")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

print("\n--- INPUTS ---")
for item in session.get_inputs():
    print(f"Name: {item.name}")
    print(f"Shape: {item.shape}")
    print(f"Type: {item.type}")

print("\n--- OUTPUTS ---")
for item in session.get_outputs():
    print(f"Name: {item.name}")
    print(f"Shape: {item.shape}")
    print(f"Type: {item.type}")

print("\n--- MODEL METADATA ---")
metadata = session.get_modelmeta()
print(f"Producer: {metadata.producer_name}")
print(f"Graph name: {metadata.graph_name}")
print(f"Description: {metadata.description}")
print(f"Domain: {metadata.domain}")
print(f"Version: {metadata.version}")

print("\n--- CUSTOM METADATA ---")
custom = metadata.custom_metadata_map
if custom:
    for key, value in custom.items():
        print(f"{key}: {value}")
else:
    print("No custom metadata found.")

print("\nInspection completed.")
