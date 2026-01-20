Your current Colab setup will not automatically save and resume training unless you explicitly implement checkpointing and resume logic in your code. Most standard PyTorch training scripts require you to add this yourself.

**What you need for robust save/resume:**
1. Save model weights, optimizer state, and epoch/counter at regular intervals (e.g., every N epochs or minutes).
2. On notebook restart, load the latest checkpoint and resume training from the saved state.

**How to implement:**
- Use torch.save() to save a dict with model.state_dict(), optimizer.state_dict(), and epoch.
- Use torch.load() to restore them.
- Add logic at the start of your training loop to check for a checkpoint and load it if present.

**Example:**
```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # add scheduler_state_dict if used
}, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

**References:**
- PyTorch docs: "Saving and Loading Models" (https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- Mediatek-PAN-DPD and OpenDPDv2 both use checkpointing for long runs.

**Conclusion:**  
If you want to avoid losing progress in Colab, you must add explicit checkpointing and resume logic. Otherwise, your training will restart from scratch after 90 minutes. This is standard practice for any long-running deep learning job.