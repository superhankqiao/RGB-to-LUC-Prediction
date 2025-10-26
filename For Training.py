import torch
from torch import nn
import os
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, \
    auc, precision_recall_curve, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import seaborn as sns
import warnings
from torch.utils.data.sampler import WeightedRandomSampler

# Ignore UserWarning, often caused by zero-division in scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)


# --- Custom Dataset ---
class CustomDataset(Dataset):
    def __init__(self, image_dir, excel_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Excel file not found: {excel_file}")

        self.df = pd.read_excel(excel_file)
        self.valid_data = []
        # Image extension as required
        self.img_extension = '.tif'

        for idx in range(len(self.df)):
            try:
                original_img_name = str(self.df.iloc[idx, 0]).strip()
                label = self.df.iloc[idx, 1]

                if pd.isna(label):
                    print(f"Warning: Skipping empty label in Excel (Index: {idx})")
                    continue
                # Ensure label is an integer
                label = int(label)

                img_path = os.path.join(self.image_dir, original_img_name + self.img_extension)

                if os.path.isfile(img_path):
                    self.valid_data.append((img_path, label, original_img_name))
                else:
                    print(f"Warning: Image file not found: {img_path}")
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid row in Excel (Index: {idx}), Error: {e}")

        if not self.valid_data:
            raise FileNotFoundError(f"No valid {self.img_extension} image files found in {self.image_dir}. "
                                   f"Check your Excel file and image folder.")

        print(f"Found {len(self.valid_data)} valid image files")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        img_path, label, img_name = self.valid_data[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, IOError) as e:
            print(f"Warning: Cannot read image file {img_path}, Error: {e}. This file will be skipped.")
            return None  # Return None, DataLoader will ignore it

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

    def get_labels(self):
        # Helper function to get all labels for stratified splitting
        return [item[1] for item in self.valid_data]


# --- New Attention & Fusion Modules (Requirement 4) ---

class SpatialAttention(nn.Module):
    """ Spatial Attention Module (CBAM) """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Ensure padding is (kernel_size - 1) // 2
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F):
        # AvgPool and MaxPool along the channel dimension
        avg_pool = torch.mean(F, dim=1, keepdim=True)
        max_pool, _ = torch.max(F, dim=1, keepdim=True)
        
        # Concatenate
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # 7x7 Convolution and Sigmoid
        A_s = self.sigmoid(self.conv(x))
        
        # Apply attention (F' = A_s * F)
        return F * A_s.expand_as(F)

class ChannelAttention(nn.Module):
    """ Channel Attention Module (SE-Net) """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            # W1 in Eq (8)
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            # W2 in Eq (8)
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, F):
        # GAP(F)
        x = self.gap(F)
        
        # W2 * ReLU(W1 * GAP(F))
        x = self.mlp(x)
        
        # A_c = sigma(...)
        A_c = self.sigmoid(x)
        
        # Apply attention (F' = A_c * F)
        return F * A_c.expand_as(F)

class MultiScaleFusion(nn.Module):
    """ Multi-scale Feature Fusion Module """
    def __init__(self, in_channels, out_channels_per_branch=128):
        super(MultiScaleFusion, self).__init__()
        
        # Calculate total output channels from branches
        total_out_channels = out_channels_per_branch * 3
        
        # 1x1 Convolution branch
        self.conv1 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=1)
        
        # 3x3 Convolution branch
        self.conv3 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=3, padding=1)
        
        # 5x5 Convolution branch
        self.conv5 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=5, padding=2)
        
        # 1x1 Convolution for dimension reduction (Eq 7)
        self.conv_1x1_final = nn.Conv2d(total_out_channels, in_channels, kernel_size=1)
        
        self.relu = nn.ReLU()

    def forward(self, F):
        # Apply three parallel convolutions
        F1 = self.relu(self.conv1(F))
        F3 = self.relu(self.conv3(F))
        F5 = self.relu(self.conv5(F))
        
        # Concatenate along the channel dimension
        x = torch.cat([F1, F3, F5], dim=1)
        
        # Final 1x1 convolution for fusion and dimension reduction
        F_multi = self.relu(self.conv_1x1_final(x))
        
        return F_multi


# --- Modified Model Definition ---
class EnhancedVGG16(nn.Module):
    """
    Enhanced VGG16 with Spatial Attention, Channel Attention,
    and Multi-Scale Fusion, following Requirement 4 (Eq 10).
    """
    def __init__(self, num_classes=1000):
        super(EnhancedVGG16, self).__init__()
        # Load pre-trained VGG16 (on ImageNet)
        base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # f(I; theta_f): Base convolutional layers
        self.features = base_model.features
        
        # Get channel dimension after features
        # VGG16's last feature map has 512 channels
        feature_channels = 512 
        
        # h_spatial
        self.spatial_att = SpatialAttention(kernel_size=7)
        
        # h_channel
        self.channel_att = ChannelAttention(in_channels=feature_channels, reduction_ratio=16)
        
        # h_fusion
        # We need to decide the output channels for fusion branches. Let's use 128.
        # 128 * 3 = 384. Final 1x1 conv will reduce 384 -> 512.
        self.fusion = MultiScaleFusion(in_channels=feature_channels, out_channels_per_branch=128)
        
        # Standard VGG classifier part
        self.avgpool = base_model.avgpool
        
        # g(...; theta_g): Final classification layers
        self.classifier = base_model.classifier
        
        # --- Transfer Learning Setup (Requirement 3) ---
        
        # Stage 1: Freeze convolutional layers
        # Note: self.features is frozen, but new modules (spatial_att, channel_att, fusion)
        # and the classifier are trainable by default.
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Replace the final layer for our number of classes
        # VGG16 classifier[6] is the last Linear layer
        in_features = self.classifier[6].in_features
        # The new classifier[6] will have requires_grad=True by default
        self.classifier[6] = nn.Linear(in_features, num_classes)
        # Dropout(0.5) is already present in VGG classifier (at index 2 and 5)
        # We can also explicitly set the dropout rate from the table
        self.classifier[2] = nn.Dropout(p=0.5) # Default is 0.5
        self.classifier[5] = nn.Dropout(p=0.5) # Default is 0.5


    def forward(self, x):
        # f(I)
        x = self.features(x)
        
        # h_spatial(f(I))
        x = self.spatial_att(x)
        
        # h_channel(h_spatial(f(I)))
        x = self.channel_att(x)
        
        # h_fusion(h_channel(h_spatial(f(I))))
        x = self.fusion(x)
        
        # Pass through classifier head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def unfreeze_features(self):
        """ Helper function for Stage 2 training """
        print("Unfreezing base model (features) parameters.")
        for param in self.features.parameters():
            param.requires_grad = True


# --- Train and Validation Functions ---
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data in dataloader:
        # Ignore invalid data
        if data is None or not data[0].size(0):
            continue
        x, y, _ = data
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(dataloader), correct / total if total > 0 else 0


def val(dataloader, model, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels, all_names = [], [], []
    with torch.no_grad():
        for data in dataloader:
            # Ignore invalid data
            if data is None or not data[0].size(0):
                continue
            x, y, names = data
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_names.extend(names)
    return total_loss / len(dataloader) if len(
        dataloader) > 0 else 0, correct / total if total > 0 else 0, all_probs, all_labels, all_names


# --- Plotting and Evaluation Functions ---
def matplot_loss(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='best')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Train and Validation Loss Curve")
    plt.savefig(save_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()


def matplot_acc(train_acc, val_acc, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title("Train and Validation Accuracy Curve")
    plt.savefig(save_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()


# --- Main Processing Function ---
def process_folder(folder_path, excel_path):
    print(f"\n--- Processing Folder: {os.path.basename(folder_path)} ---")

    # --- Parameter Setup (Requirement 3) ---
    
    # Data Augmentation (for training)
    train_transform = transforms.Compose([
        transforms.Resize((51, 51)),
        # Data Augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(15)], p=0.33),
        transforms.RandomApply([transforms.ColorJitter(brightness=(0.8, 1.2))], p=0.5),
        # ToTensor and Normalize
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation/Test Transform (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((51, 51)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Parameters
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    STAGE1_EPOCHS = 20
    EARLY_STOP_PATIENCE = 15
    # Optimizer Params
    LR_STAGE1 = 0.001
    LR_STAGE2 = 0.0001
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    WEIGHT_DECAY = 1e-4
    # Scheduler Params
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    try:
        # Load dataset once to get labels for splitting
        dataset_for_split = CustomDataset(image_dir=folder_path, excel_file=excel_path, transform=val_test_transform)
    except FileNotFoundError as e:
        print(f"Skipping folder {os.path.basename(folder_path)}, Reason: {e}")
        return

    labels = dataset_for_split.get_labels()
    num_classes = len(set(labels))
    indices = list(range(len(dataset_for_split)))

    print(f"Detected {num_classes} classes")

    if num_classes < 2:
        print(f"Warning: Folder {os.path.basename(folder_path)} only contains {num_classes} class(es). Skipping.")
        return

    # Create a result folder named "one_time"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(folder_path, f"one_{current_time}")
    os.makedirs(result_folder, exist_ok=True)
    
    save_model_dir = os.path.join(result_folder, "save_model")
    os.makedirs(save_model_dir, exist_ok=True)


    # Data Split (Requirement 3: 80% Train, 10% Val, 10% Test)
    # Using Stratified Sampling
    try:
        # Split 80% train, 20% temp (val+test)
        train_indices, temp_indices, y_train, y_temp = train_test_split(
            indices, labels, test_size=0.2, stratify=labels, random_state=42
        )
        # Split 20% temp into 10% val, 10% test (0.5 * 0.2 = 0.1)
        val_indices, test_indices, y_val, y_test = train_test_split(
            temp_indices, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
    except ValueError as e:
        print(f"Error during stratified split (likely too few samples in a class): {e}. Skipping folder.")
        return

    # Create dataset objects with respective transforms
    train_dataset = Subset(
        CustomDataset(image_dir=folder_path, excel_file=excel_path, transform=train_transform),
        train_indices
    )
    val_dataset = Subset(
        CustomDataset(image_dir=folder_path, excel_file=excel_path, transform=val_test_transform),
        val_indices
    )
    test_dataset = Subset(
        CustomDataset(image_dir=folder_path, excel_file=excel_path, transform=val_test_transform),
        test_indices
    )

    print(f"Total: {len(dataset_for_split)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Handle sample imbalance (WeightedRandomSampler from original code)
    train_labels = [labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / (class_counts + 1e-6) # Add epsilon to avoid division by zero
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Filter out None values returned by __getitem__
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else (torch.empty(0), torch.empty(0), [])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, sampler=sampler,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Use the new enhanced model
    model = EnhancedVGG16(num_classes=num_classes).to(device)

    # Loss function (Categorical CrossEntropy)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_path = os.path.join(save_model_dir, "best_model.pth")

    # --- Training Stage 1 (Requirement 3: Freeze features, 20 epochs) ---
    print(f"\n--- Starting Training Stage 1 (Epoch 1 to {STAGE1_EPOCHS}) ---")
    print("Training: Classifier layers + new modules")
    
    # Optimizer for Stage 1 (only classifier and new modules)
    # model.features is frozen, so optimizer will only update non-frozen params
    optimizer_stage1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE1,
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=WEIGHT_DECAY
    )
    
    # LR Scheduler for Stage 1
    scheduler_stage1 = lr_scheduler.ReduceLROnPlateau(
        optimizer_stage1, mode='min', factor=SCHEDULER_FACTOR, 
        patience=SCHEDULER_PATIENCE, min_lr=MIN_LR, verbose=True
    )

    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer_stage1, device)
        val_loss, val_acc, _, _, _ = val(val_dataloader, model, loss_fn, device)

        print(f'Epoch [{epoch + 1:03d}/{MAX_EPOCHS:03d}] [Stage 1] | '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler_stage1.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'↳ New best model saved! Val Loss: {best_val_loss:.4f}')

    # --- Training Stage 2 (Requirement 3: Unfreeze, fine-tune) ---
    print(f"\n--- Starting Training Stage 2 (Epoch {STAGE1_EPOCHS + 1} to {MAX_EPOCHS}) ---")
    model.unfreeze_features() # Unfreeze base model

    # Optimizer for Stage 2 (all parameters) with new LR
    optimizer_stage2 = torch.optim.Adam(
        model.parameters(),
        lr=LR_STAGE2, # Use stage 2 learning rate
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=WEIGHT_DECAY
    )
    
    # LR Scheduler for Stage 2
    scheduler_stage2 = lr_scheduler.ReduceLROnPlateau(
        optimizer_stage2, mode='min', factor=SCHEDULER_FACTOR, 
        patience=SCHEDULER_PATIENCE, min_lr=MIN_LR, verbose=True
    )
    
    # Reset best val loss and early stopping counter
    best_val_loss = float('inf') # Reset to ensure we save the best model from stage 2
    early_stopping_counter = 0
    
    # Load the best model from stage 1 before starting stage 2
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model from Stage 1 to continue fine-tuning.")


    for epoch in range(STAGE1_EPOCHS, MAX_EPOCHS):
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer_stage2, device)
        val_loss, val_acc, _, _, _ = val(val_dataloader, model, loss_fn, device)

        print(f'Epoch [{epoch + 1:03d}/{MAX_EPOCHS:03d}] [Stage 2] | '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler_stage2.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'↳ New best model saved! Val Loss: {best_val_loss:.4f}')
            early_stopping_counter = 0 # Reset counter
        else:
            early_stopping_counter += 1
            print(f'↳ No improvement. Early stopping counter: {early_stopping_counter}/{EARLY_STOP_PATIENCE}')

        # Early Stopping
        if early_stopping_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1} due to no improvement in validation loss.")
            break

    # --- Testing Phase ---
    if not os.path.exists(best_model_path):
        print("No best model file found, cannot perform testing.")
        return

    model.load_state_dict(torch.load(best_model_path))

    print("\nStarting testing...")
    test_loss, test_acc, test_probs, test_labels, test_names = val(test_dataloader, model, loss_fn, device)
    test_preds = np.argmax(np.array(test_probs), axis=1)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    # Calculate detailed metrics
    cm = confusion_matrix(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(test_labels, test_preds)

    y_prob = np.array(test_probs)
    roc_auc = "N/A"
    pr_auc = "N/A"
    fpr, tpr, pr_precision, pr_recall = None, None, None, None
    
    if num_classes == 2:
        try:
            fpr, tpr, _ = roc_curve(test_labels, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            pr_precision, pr_recall, _ = precision_recall_curve(test_labels, y_prob[:, 1])
            pr_auc = average_precision_score(test_labels, y_prob[:, 1])
        except Exception as e:
            print(f"Could not calculate ROC/PR AUC: {e}")
            
    class_names = [f'Class_{i}' for i in sorted(list(set(test_labels)))]

    # Plot and save images in SVG format
    matplot_loss(train_losses, val_losses, os.path.join(result_folder, "loss_curve.svg"))
    matplot_acc(train_accs, val_accs, os.path.join(result_folder, "acc_curve.svg"))
    plot_confusion_matrix(cm, class_names, os.path.join(result_folder, "confusion_matrix.svg"))

    if num_classes == 2 and fpr is not None:
        plot_roc_curve(fpr, tpr, roc_auc, os.path.join(result_folder, "roc_curve.svg"))

    # Write all performance metrics to metrics.txt
    with open(os.path.join(result_folder, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'--- Best Model Performance Metrics (Folder: {os.path.basename(folder_path)}) ---\n')
        f.write('================================================\n\n')
        f.write(f'Final Test Loss: {test_loss:.4f}\n')
        f.write(f'Final Test Accuracy: {test_acc:.4f}\n')
        f.write(f'Weighted Precision: {precision:.4f}\n')
        f.write(f'Weighted Recall: {recall:.4f}\n')
        f.write(f'F1-Score (Weighted): {f1:.4f}\n')
        f.write(f'Kappa Coefficient: {kappa:.4f}\n')
        if num_classes == 2:
            f.write(f'ROC Curve AUC: {roc_auc:.4f}\n')
            f.write(f'PR Curve AUC: {pr_auc:.4f}\n')
        else:
            f.write(f'ROC Curve AUC: {roc_auc} (multi-class)\n')
            f.write(f'PR Curve AUC: {pr_auc} (multi-class)\n')
        f.write('\nConfusion Matrix:\n')
        f.write(str(cm) + '\n\n')
        f.write('Classification Report (Precision, Recall, F1-Score):\n')
        f.write(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    # Save final predictions to XLSX
    results_df = pd.DataFrame({
        'Image_Name': test_names,
        'True_Label': test_labels,
        'Predicted_Label': test_preds,
        'Prediction_Probability': [max(probs) for probs in test_probs]
    })
    results_df.to_excel(os.path.join(result_folder, 'test_predictions.xlsx'), index=False)

    print(f"Folder {os.path.basename(folder_path)} processing complete. Results saved in: {result_folder}")


# --- Main Program Entry ---
if __name__ == '__main__':
    # --- IMPORTANT: Please modify the following paths ---

    # ROOT_FOLDER: The root directory containing all training subfolders (e.g., model1, model2).
    ROOT_FOLDER = 'path/to/your/training_data_root_folder' 
    
    # MAIN_EXCEL_FILE_NAME: The name of the main Excel file containing image names and labels.
    # This file should be located directly inside the ROOT_FOLDER.
    MAIN_EXCEL_FILE_NAME = 'Label.xls'
    MAIN_EXCEL_FILE = os.path.join(ROOT_FOLDER, MAIN_EXCEL_FILE_NAME)

    # Check if the paths have been set correctly
    if 'path/to/your' in ROOT_FOLDER:
        print(f"Error: Please update the 'ROOT_FOLDER' variable in the script to your actual training data directory.")
    elif not os.path.exists(ROOT_FOLDER):
        print(f"Error: Root folder {ROOT_FOLDER} does not exist. Please check the path.")
    elif not os.path.exists(MAIN_EXCEL_FILE):
        print(f"Error: Main Excel file {MAIN_EXCEL_FILE} does not exist. Please check the path.")
    else:
        # Get all subdirectories
        subfolders = [os.path.join(ROOT_FOLDER, d) for d in os.listdir(ROOT_FOLDER) if
                      os.path.isdir(os.path.join(ROOT_FOLDER, d))]

        if not subfolders:
            print(f"No subfolders found in {ROOT_FOLDER}.")
        else:
            print(f"Found the following subfolders to process in {ROOT_FOLDER}: {[os.path.basename(f) for f in subfolders if not os.path.basename(f).startswith('one_')]}")
            for folder in subfolders:
                # Avoid processing result folders ('one_...') themselves
                if not os.path.basename(folder).startswith('one_'):
                    process_folder(folder, MAIN_EXCEL_FILE)