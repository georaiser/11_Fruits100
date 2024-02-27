""" training and testing a PyTorch model """

import os
import torch

from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d%H%M')

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    
    """ Trains a PyTorch model for a single epoch """
    
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values to zero
    train_loss, train_acc = 0, 0

    # Loop through dataloader batches
    for batch, (X, y) in tqdm(enumerate(dataloader), desc='Training Step', leave=False) :
        # Send data to target device
        X, y = X.to(device), y.to(device) 

        # Forward pass
        y_pred = model(X)

        # Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric classification across batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def validation_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    
    """ Tests a PyTorch model for a single epoch """
    
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values to zero
    validation_loss, validation_acc = 0, 0

    # Turn on inference context manager
    with torch.no_grad():
        # Loop through DataLoader batches
        for batch, (X, y) in tqdm(enumerate(dataloader), desc='Validation Step', leave=False):
            # Send data to target device
            X, y = X.to(device), y.to(device) 

            # Forward pass
            validation_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(validation_pred_logits, y)
            validation_loss += loss.item()

            # Calculate and accumulate accuracy
            validation_pred_labels = validation_pred_logits.argmax(dim=1)
            validation_acc += ((validation_pred_labels == y).sum().item()/len(validation_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    validation_loss = validation_loss / len(dataloader)
    validation_acc = validation_acc / len(dataloader)
    return validation_loss, validation_acc

def run_model(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          validation_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler:torch.optim,
          early_stopping,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
    
    writer = SummaryWriter(f'runs/training_{model.__class__.__name__}_{timestamp}')
    
    """ Trains and tests a PyTorch model """
    
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "validation_loss": [],
               "validation_acc": []
    }
    
    # Training and validation loop
    best_vloss = float('inf')
    
    for epoch in tqdm(range(epochs)):
        
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        validation_loss, validation_acc = validation_step(model=model,
                                        dataloader=validation_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        scheduler.step(validation_loss)

        # Print out running process
        print(f"Epoch: {epoch+1} | "
              f"lr: {optimizer.param_groups[0]['lr']} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"validation_loss: {validation_loss:.4f} | "
              f"validation_acc: {validation_acc:.4f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["validation_loss"].append(validation_loss)
        results["validation_acc"].append(validation_acc)
        
        # for both training and validation
        # Add results to SummaryWriter
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "val_loss": validation_loss},
                           global_step=epoch)
        
        writer.add_scalars(main_tag="Accuracy", 
                           tag_scalar_dict={"train_acc": train_acc,
                                            "val_acc": validation_acc}, 
                           global_step=epoch+1)
        
        writer.flush()
        
        if not os.path.exists("models"):
            os.makedirs("models/", exist_ok=True)
            
        # Track best performance, and save the model's state
        if validation_loss < best_vloss:
            best_vloss = validation_loss
            model_path_state_dict = f"models/model_state_dict_{model.__class__.__name__}_{timestamp}.pth"
            torch.save(model.state_dict(), model_path_state_dict)
            model_path_full = f"models/model_full_{model.__class__.__name__}_{timestamp}.pth"
            torch.save(model, model_path_full)
            print(f"model saved at epoch: {epoch+1}")
            
        # early stopping
        early_stopping(validation_loss, validation_acc)  # Monitor validation accuracy
        if early_stopping.early_stop:
          print(f"early stopping at epoch: {epoch}")
          break
            
    writer.close()

    # Return results at the end of epochs
    return results
