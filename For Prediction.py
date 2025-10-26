import torch
from torch import nn
import os
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, UnidentifiedImageError
import numpy as np
import warnings

# Ignore UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


# --- Model Definition (MUST MATCH TRAINING CODE) ---
# This entire model definition (including attention modules)
# is required to load the .pth file.

class SpatialAttention(nn.Module):
    """ Spatial Attention Module (CBAM) """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F):
        avg_pool = torch.mean(F, dim=1, keepdim=True)
        max_pool, _ = torch.max(F, dim=1, keepdim=True)
        x = torch.cat([avg_pool, max_pool], dim=1)
        A_s = self.sigmoid(self.conv(x))
        return F * A_s.expand_as(F)

class ChannelAttention(nn.Module):
    """ Channel Attention Module (SE-Net) """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, F):
        x = self.gap(F)
        x = self.mlp(x)
        A_c = self.sigmoid(x)
        return F * A_c.expand_as(F)

class MultiScaleFusion(nn.Module):
    """ Multi-scale Feature Fusion Module """
    def __init__(self, in_channels, out_channels_per_branch=128):
        super(MultiScaleFusion, self).__init__()
        total_out_channels = out_channels_per_branch * 3
        self.conv1 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=5, padding=2)
        self.conv_1x1_final = nn.Conv2d(total_out_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, F):
        F1 = self.relu(self.conv1(F))
        F3 = self.relu(self.conv3(F))
        F5 = self.relu(self.conv5(F))
        x = torch.cat([F1, F3, F5], dim=1)
        F_multi = self.relu(self.conv_1x1_final(x))
        return F_multi

class EnhancedVGG16(nn.Module):
    """
    Enhanced VGG16 definition. Must be identical to the one used in training.
    """
    def __init__(self, num_classes=1000):
        super(EnhancedVGG16, self).__init__()
        base_model = models.vgg16(weights=None) # No need for weights at inference
        
        self.features = base_model.features
        feature_channels = 512 
        self.spatial_att = SpatialAttention(kernel_size=7)
        self.channel_att = ChannelAttention(in_channels=feature_channels, reduction_ratio=16)
        self.fusion = MultiScaleFusion(in_channels=feature_channels, out_channels_per_branch=128)
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        
        # Set dropout (same as training)
        self.classifier[2] = nn.Dropout(p=0.5)
        self.classifier[5] = nn.Dropout(p=0.5)
        
        # Replace the final layer
        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.spatial_att(x)
        x = self.channel_att(x)
        x = self.fusion(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- Prediction-Only Dataset ---
# This class loads images from a folder without needing labels
class PredictDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        # Find all .tif files in the directory
        for img_name in os.listdir(image_dir):
            if img_name.lower().endswith('.tif'):
                self.image_paths.append(os.path.join(image_dir, img_name))

        if not self.image_paths:
            raise FileNotFoundError(f"No .tif image files found in {self.image_dir}.")

        print(f"Found {len(self.image_paths)} image file(s) for prediction.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Get filename without extension
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        try:
            # Try to open and convert image
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, IOError) as e:
            print(f"Warning: Cannot read image file {img_path}, Error: {e}. This file will be skipped.")
            return None

        if self.transform:
            image = self.transform(image)

        # Return image tensor and filename
        return image, img_name


# --- Ensemble Prediction Function (Requirement 5) ---
def ensemble_predict(model_ensemble_folder, predict_data_folder, output_excel, num_classes, threshold):
    print(f"\n--- Starting Ensemble Prediction ---")
    print(f"Model Folder: {model_ensemble_folder}")
    print(f"Data Folder: {predict_data_folder}")

    # Define the same image preprocessing as validation/testing
    transform = transforms.Compose([
        transforms.Resize((51, 51)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load Prediction Data
    try:
        dataset = PredictDataset(image_dir=predict_data_folder, transform=transform)
    except FileNotFoundError as e:
        print(f"Error: {e}. Aborting prediction.")
        return

    # collate_fn to handle None values
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else None

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Load All Candidate Models
    model_paths = []
    # Find all "best_model.pth" in the ensemble folder
    for root, dirs, files in os.walk(model_ensemble_folder):
        if 'best_model.pth' in files:
            model_paths.append(os.path.join(root, 'best_model.pth'))
    
    if not model_paths:
        print(f"Error: No 'best_model.pth' files found in {model_ensemble_folder} or its subdirectories.")
        return

    N = len(model_paths)
    print(f"Found {N} candidate models for the ensemble.")
    
    models = []
    for path in model_paths:
        try:
            model = EnhancedVGG16(num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval() # Set model to evaluation mode
            models.append(model)
            print(f"Successfully loaded model: {path}")
        except Exception as e:
            print(f"Warning: Failed to load model {path}. Error: {e}. Skipping this model.")
            
    N = len(models) # Update N in case some models failed to load
    if N == 0:
        print("Error: No models were successfully loaded. Aborting.")
        return
    print(f"--- {N} models loaded successfully. Starting prediction. ---")


    all_names = []
    all_final_preds = []
    all_consistency_ratios = []
    
    # 3. Perform Ensemble Prediction
    with torch.no_grad():
        for data in dataloader:
            if data is None:
                continue
            
            x, names = data
            x = x.to(device)
            B = x.size(0) # Current batch size
            
            # Vote array for the batch: shape (Batch_Size, Num_Classes)
            batch_votes = np.zeros((B, num_classes))
            
            # Get vote from each model
            for model in models:
                output = model(x)
                # Get predicted class index
                preds = torch.argmax(output, dim=1)
                
                # Add votes
                for i, p in enumerate(preds.cpu().numpy()):
                    batch_votes[i, p] += 1
            
            # 4. Apply Threshold Voting (Eq 11)
            
            # r_k = V_k / N
            vote_ratios = batch_votes / N
            
            # Get consistency (r_max)
            consistency = np.max(vote_ratios, axis=1)
            
            # Get the class with the most votes
            predicted_class = np.argmax(vote_ratios, axis=1)
            
            # Apply threshold theta (Requirement 5)
            final_predictions = []
            for i in range(B):
                if consistency[i] > threshold:
                    # Mark as the predicted class
                    final_predictions.append(str(predicted_class[i]))
                else:
                    # Mark as "Uncertain"
                    final_predictions.append("Uncertain")
            
            all_names.extend(names)
            all_final_preds.extend(final_predictions)
            all_consistency_ratios.extend(consistency)

    # 5. Save Results
    # Output includes: ① Main prediction, ② Consistency, ③ Uncertain mark
    results_df = pd.DataFrame({
        'Image_Name': all_names,
        'Predicted_Class': all_final_preds,
        'Model_Consistency_Ratio': all_consistency_ratios
    })

    results_df.to_excel(output_excel, index=False)
    print(f"\nEnsemble prediction complete. Results saved to: {output_excel}")


# --- Main Program Entry ---
if __name__ == '__main__':
    # --- IMPORTANT: Please modify the following paths and parameters ---
    
    # ENSEMBLE_MODEL_FOLDER: The parent directory containing all candidate model folders.
    # (e.g., a folder that contains model1/one_.../, model2/one_.../, etc.)
    ENSEMBLE_MODEL_FOLDER = 'path/to/your/ensemble_model_folder'

    # PREDICT_DATA_FOLDER: The directory containing the new .tif images to be predicted.
    PREDICT_DATA_FOLDER = 'path/to/your/prediction_data_folder'

    # OUTPUT_EXCEL_FILE: The full path, including the filename, for the output Excel file.
    OUTPUT_EXCEL_FILE = 'path/to/your/output_directory/ensemble_predictions.xlsx'
    
    # --- Parameters from Requirement 5 ---
    
    # NUM_CLASSES: The total number of classes the models were trained on. This must match.
    NUM_CLASSES = 3 
    
    # VOTING_THRESHOLD: The consistency threshold (theta) for the ensemble vote.
    VOTING_THRESHOLD = 0.6

    # Check if paths have been modified
    if 'path/to/your' in ENSEMBLE_MODEL_FOLDER or \
       'path/to/your' in PREDICT_DATA_FOLDER or \
       'path/to/your' in OUTPUT_EXCEL_FILE:
        print("Error: Please update 'ENSEMBLE_MODEL_FOLDER', 'PREDICT_DATA_FOLDER', and 'OUTPUT_EXCEL_FILE' to your actual paths.")
    
    # Check if paths exist
    elif not os.path.exists(ENSEMBLE_MODEL_FOLDER):
        print(f"Error: Ensemble model folder not found: {ENSEMBLE_MODEL_FOLDER}")
    elif not os.path.exists(PREDICT_DATA_FOLDER):
        print(f"Error: Prediction data folder not found: {PREDICT_DATA_FOLDER}")
    else:
        # Ensure the output directory exists
        output_dir = os.path.dirname(OUTPUT_EXCEL_FILE)
        if output_dir and not os.path.exists(output_dir):
            print(f"Warning: Output directory {output_dir} does not exist. Attempting to create it.")
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                print(f"Error: Could not create output directory {output_dir}. Error: {e}")
                # Exit or handle the error as needed
    
        ensemble_predict(
            model_ensemble_folder=ENSEMBLE_MODEL_FOLDER,
            predict_data_folder=PREDICT_DATA_FOLDER,
            output_excel=OUTPUT_EXCEL_FILE,
            num_classes=NUM_CLASSES,
            threshold=VOTING_THRESHOLD
        )