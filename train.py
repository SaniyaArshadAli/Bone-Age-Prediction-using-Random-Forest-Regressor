import os
import cv2
import numpy as np
import pandas as pd
import kagglehub
import joblib
from PIL import Image
from skimage import feature, measure
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm # For a beautiful progress bar in the terminal

def process_single_image(img_path, is_male):
    """Worker function for parallel processing."""
    try:
        # Load and Grayscale
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        
        # CLAHE Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_array)
        processed = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Morphology
        thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        labels = measure.label(thresh)
        props = measure.regionprops(labels)
        
        if props:
            main_bone = sorted(props, key=lambda x: x.area, reverse=True)[0]
            area, peri = main_bone.area, main_bone.perimeter
            ecc, sol = main_bone.eccentricity, main_bone.solidity
        else:
            area = peri = ecc = sol = 0

        # Texture
        processed_8bit = (processed / (processed.max() + 1e-5) * 255).astype(np.uint8)
        glcm = feature.graycomatrix(processed_8bit, [1], [0, np.pi/4], 256, symmetric=True, normed=True)
        cont = feature.graycoprops(glcm, 'contrast')[0, 0]
        corr = feature.graycoprops(glcm, 'correlation')[0, 0]

        return [area, peri, ecc, sol, cont, corr, np.mean(processed), 1 if is_male else 0]
    except Exception as e:
        return None

def main():
    print("🚀 Initiating Full Dataset Download (RSNA Bone Age)...")
    dataset_path = kagglehub.dataset_download("kmader/rsna-bone-age")
    
    csv_path = os.path.join(dataset_path, 'boneage-training-dataset.csv')
    img_dir = os.path.join(dataset_path, 'boneage-training-dataset', 'boneage-training-dataset')
    
    df = pd.read_csv(csv_path)
    print(f"📊 Found {len(df)} patient records. Beginning feature extraction...")
    
    # Optional: Limit to first 2000 for faster training, or remove `.head()` for ALL data
    df = df.head(2000) 
    
    # Prepare arguments for parallel processing
    tasks = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, f"{row['id']}.png")
        if os.path.exists(img_path):
            tasks.append((img_path, row['male']))
            labels.append(row['boneage'])

    # Run extraction on all CPU cores
    print("⚙️ Extracting Bio-features (This may take a while...)")
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(
        delayed(process_single_image)(path, sex) for path, sex in tqdm(tasks)
    )

    # Clean up any failed image reads
    valid_data = [(res, labels[i]) for i, res in enumerate(results) if res is not None]
    X_raw = [item[0] for item in valid_data]
    y = [item[1] for item in valid_data]

    print("🧠 Training Random Forest Regressor...")
    X = pd.DataFrame(X_raw, columns=["Area", "Perimeter", "Eccentricity", "Solidity", "Contrast", "Correlation", "Mean_Intensity", "Is_Male"])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Robust Model Configuration
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    
    # Save the artifacts
    joblib.dump(model, 'production_model.pkl')
    joblib.dump(scaler, 'production_scaler.pkl')
    print("✅ Training Complete! Artifacts saved as 'production_model.pkl'. You may now run app.py.")

if __name__ == "__main__":
    main()