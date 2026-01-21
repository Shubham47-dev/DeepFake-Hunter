import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import your modules
# NOTE: Make sure you run this script from the root directory (DeepFake-Hunter/)
from models.dual_fusion import DualBranchDetector
from training.dataset_loader import DualStreamDataset
from training.loss import DeepFakeLoss

# --- CONFIGURATION --- #
DATA_DIR = "data/train"  # Folder containing 'Real' and 'Fake' folders
BATCH_SIZE = 16          # Reduce to 8 if you run out of GPU memory
EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print(f"🚀 Training on {DEVICE}...")
    
    # 1. Prepare Data
    full_dataset = DualStreamDataset(root_dir=DATA_DIR)
    
    # Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 Data Loaded: {len(train_dataset)} Train images, {len(val_dataset)} Val images")

    # 2. Initialize Model, Loss, Optimizer
    model = DualBranchDetector().to(DEVICE)
    criterion = DeepFakeLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for rgb, freq, labels in loop:
            rgb, freq, labels = rgb.to(DEVICE), freq.to(DEVICE), labels.to(DEVICE)
            
            # Zero Gradients
            optimizer.zero_grad()
            
            # Forward Pass (The Dual Branch Magic)
            outputs = model(rgb, freq)
            loss = criterion(outputs, labels)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # 4. Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for rgb, freq, labels in val_loader:
                rgb, freq, labels = rgb.to(DEVICE), freq.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(rgb, freq)
                
                # Convert logit to probability using Sigmoid
                probs = torch.sigmoid(outputs).view(-1)
                predicted = (probs > 0.5).float()
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"✅ Epoch {epoch+1} Results -> Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        # 5. Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 Model Saved!")

if __name__ == "__main__":
    train()