import json
import pickle
import shutil
from pathlib import Path
import filecmp
from functools import lru_cache
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ACTIVE_FILE = MODELS_DIR / "active_model.json"

def register_model(model_file, version):

    model_file = Path(model_file)
    metadata_file = model_file.with_name(f"{model_file.stem}_metadata.json")
    model_dir = MODELS_DIR / version

    if not model_file.is_file():
        raise FileNotFoundError(f"Model not found: {model_file}")

    if not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")

    if model_dir.exists():
        raise FileExistsError(f"Model {version} already exists")

    # Read metadata and add the version number
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    metadata["version"] = version

    # Create version folder
    model_dir.mkdir(parents=True)

    # Save model and metadata
    shutil.copy2(model_file, model_dir / "model.pkl")

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Registered model {version}")


def activate_model(version):

    model_path = MODELS_DIR / version / "model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model {version} does not exist")

    with open(ACTIVE_FILE, "w") as f:
        json.dump({"active_version": version}, f, indent=4)

    print(f"Active model changed to {version}")

# loads and caches the current model version
@lru_cache(maxsize=4)
def load_model(version):
    model_path = MODELS_DIR / version / "model.pkl"

    with open(model_path, "rb") as f:
        return pickle.load(f)

# retrieves current model version
def get_model():
    with open(MODELS_DIR / "active_model.json", "r") as f:
        version = json.load(f)["active_version"]

    with open(MODELS_DIR / version / "metadata.json", "r") as f:
        metadata = json.load(f)

    return load_model(version), metadata, version

# lists all registered models with their metadata and active status
def list_models():
    with open(ACTIVE_FILE, "r") as f:
        active_version = json.load(f)["active_version"]

    models = []

    for model_dir in sorted(MODELS_DIR.glob("v*")):
        metadata_file = model_dir / "metadata.json"

        if not metadata_file.is_file():
            continue

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        metadata["active"] = model_dir.name == active_version
        models.append(metadata)

    return models
