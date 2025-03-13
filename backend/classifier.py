import os
import cv2
import faiss
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up directories
IMAGE_FOLDER = "stored_images"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Mount static files to serve images
app.mount("/stored_images", StaticFiles(directory=IMAGE_FOLDER), name="stored_images")

# Load pre-trained ResNet50 model for feature extraction
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# FAISS Index setup
d = 2048  # Feature vector size (ResNet50 output)
index = faiss.IndexFlatL2(d)

# Load stored embeddings (if available)
embedding_file = "image_embeddings.pkl"
if os.path.exists(embedding_file):
    with open(embedding_file, "rb") as f:
        image_db = pickle.load(f)
        if image_db:
            vectors = np.array(list(image_db.values()), dtype="float32")
            index.add(vectors)
else:
    image_db = {}

def extract_features(image):
    """Extract feature vector from an image using ResNet50."""
    image = image.convert("RGB")  # Convert RGBA/Grayscale to RGB
    image = image.resize((224, 224))  # Resize to model input size
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize for ResNet50
    return model.predict(img_array)[0]  # Extract features

def is_duplicate(new_vector, threshold=0.8):
    """Check if an image is a duplicate using FAISS."""
    if index.ntotal == 0:
        return False  # No images in database yet
    D, I = index.search(np.array([new_vector]).astype('float32'), k=1)
    return D[0][0] < (1 - threshold)  # FAISS uses L2 distance

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """API Endpoint to upload and check/store unique images."""
    image = Image.open(BytesIO(await file.read()))
    new_vector = extract_features(image)

    if is_duplicate(new_vector):
        return {"message": "Duplicate image detected. Not added."}

    # Save unique image
    file_path = os.path.join(IMAGE_FOLDER, file.filename)
    image.save(file_path)

    # Store embedding
    image_db[file.filename] = new_vector
    index.add(np.array([new_vector]).astype("float32"))

    # Save updated database
    with open(embedding_file, "wb") as f:
        pickle.dump(image_db, f)

    return {"message": "New unique image added.", "filename": file.filename}

@app.get("/")
def home():
    return {"message": "FAISS Image Deduplication API Running!"}

@app.get("/images/")
def get_stored_images():
    """Returns URLs of stored unique images."""
    image_urls = [f"/stored_images/{filename}" for filename in image_db.keys()]
    return {"stored_images": image_urls}

@app.delete("/delete/{filename}")
async def delete_image(filename: str):
    """API to delete an image from storage and FAISS index."""
    file_path = os.path.join(IMAGE_FOLDER, filename)

    if filename not in image_db:
        return {"error": "Image not found."}

    # Delete image file
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove from FAISS index
    del image_db[filename]
    index.reset()  # FAISS does not support direct deletion; we need to rebuild it.

    # Re-add remaining vectors
    if image_db:
        vectors = np.array(list(image_db.values()), dtype="float32")
        index.add(vectors)

    # Update stored embeddings file
    with open(embedding_file, "wb") as f:
        pickle.dump(image_db, f)

    return {"message": f"Image '{filename}' deleted successfully."}