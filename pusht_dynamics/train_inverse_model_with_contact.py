import torch
import torch.nn as nn

from datetime import datetime
from tqdm import tqdm

from datasets import InverseDynamicsContactDataset
from models import InverseDynamics
from utils import (
    make_fast_loader, make_writer, get_lr,
    setup_optimizer, split_episodes, save_topk, get_plain_state_dict
)

# --- bring your existing episode loader ---
# from pusht.datasets import load_episode_dataset
from pusht.datasets import load_episode_dataset

# ---- config ----
dataset_path = "models/pusht_cchi_v7_replay.zarr.zip"

# fix me: two important settings for data processing
use_relative_action = False
use_object_centric_frame = False
# use_relative_action = True
# use_object_centric_frame = True
detect_contact = True

epochs = 1000
batch_size = 512
lr = 1e-3
val_ratio = 0.2
relative = "re" if use_relative_action else "abs"
run_name = f"inverse_{relative}_contact_{detect_contact}_bs{batch_size}_lr{lr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# ---- device + AMP for RTX 2080 Ti (FP16) ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
amp_dtype = torch.float16
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

torch.backends.cudnn.benchmark = False  # ok for MLPs/CNNs with steady shapes

# ---- data ----
base = load_episode_dataset(dataset_path, use_relative_action=use_relative_action, use_object_centric_frame=use_object_centric_frame, detect_contact=detect_contact)
base_train, base_val = split_episodes(base, val_ratio=val_ratio, seed=42)
train_ds = InverseDynamicsContactDataset(base_train)
val_ds   = InverseDynamicsContactDataset(base_val)
train_loader = make_fast_loader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = make_fast_loader(val_ds,   batch_size=batch_size, shuffle=False)

# ---- model ----
# infer dims from one sample
s = next(iter(train_loader))
obs_dim = int(s["o_curr"].view(s["o_curr"].shape[0], -1).shape[-1])
act_dim = int(s["action"].view(s["action"].shape[0], -1).shape[-1])

model = InverseDynamics(obs_dim, act_dim).to(device)
try:
    model = torch.compile(model, mode="max-autotune")  # PyTorch 2+
except Exception:
    pass

opt = setup_optimizer(model, lr=lr)
loss_fn = nn.MSELoss()  # swap to CrossEntropy if your actions are discrete

# --- logging ---
writer = make_writer(run_name)
writer.add_text("config", f"epochs={epochs}, batch={batch_size}, lr={lr}, amp=fp16, val_ratio={val_ratio}")
run_dir = writer.log_dir
best_val = float("inf")

# ---- train ----
global_step = 0
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"[Epoch {epoch:02d}][train]", leave=False)
    for step, batch in enumerate(train_bar, 1):
        o_prev = batch["o_prev"].to(device, non_blocking=True)
        o_curr = batch["o_curr"].to(device, non_blocking=True)
        o_next = batch["o_next"].to(device, non_blocking=True)
        a_t    = batch["action"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device=="cuda")):
            pred = model(o_prev, o_curr, o_next)
            loss = loss_fn(pred, a_t)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        epoch_loss += loss.item()

        writer.add_scalar("train/train_step", loss.item(), global_step)
        writer.add_scalar("train/lr", get_lr(opt), global_step)

        global_step += 1    
        if step % 50 == 0:
            print(f"[inverse][epoch {epoch:02d} step {step:05d}] loss={epoch_loss/50:.6f}")
            epoch_loss = 0.0
    # --- validation ---
    model.eval()
    val_loss_sum, val_count = 0.0, 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device=="cuda")):
        val_bar = tqdm(val_loader, desc=f"[Epoch {epoch:02d}][val]", leave=False)
        for batch in val_bar:
            o_prev = batch["o_prev"].to(device, non_blocking=True)
            o_curr = batch["o_curr"].to(device, non_blocking=True)
            o_next = batch["o_next"].to(device, non_blocking=True)
            a_t    = batch["action"].to(device, non_blocking=True)
            pred = model(o_prev, o_curr, o_next)
            vloss = loss_fn(pred, a_t)
            bs = o_curr.shape[0]
            val_loss_sum += vloss.item() * bs
            val_count += bs
    val_loss = val_loss_sum / max(1, val_count)
    writer.add_scalar("val/loss_epoch", val_loss, epoch)
    print(f"[inverse][epoch {epoch:02d}] val_loss={val_loss:.6f}")

    # save best
    if val_loss < best_val:
        best_val = val_loss
        save_topk({"model": get_plain_state_dict(model), "epoch": epoch, "val_loss": val_loss}, run_dir)

writer.close()
print(f"Best val loss: {best_val:.6f} (checkpoint saved under {run_dir}/checkpoints/best.pt)")