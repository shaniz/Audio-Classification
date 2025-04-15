import torch
import numpy as np
import argparse
import cca_core
import pickle
import gzip
from dataloaders.datasetaug import val_dataloader
from utils import Params
import torch

##### ADD TO requirements:
# pip install scikit-learn
# pip install --no-deps cca_core


def load_checkpoint_analysis(checkpoint_path, model, optimizer=None, parallel=False):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load with map_location to CPU or current device
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if parallel:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model


# Hook class to extract layer activations during forward pass
class Hook:
    def __init__(self, model, layers):
        self.acts = {l: [] for l in layers}
        for name, mod in model.named_modules():
            # print(name)
            if name in layers:
                print(f"✅ Registered hook on: {name}")

                def hook_fn(m, i, o, n=name):
                    print(f"🔥 Activation captured for: {n}, shape: {o.shape}")
                    self.acts[n].append(o.detach().cpu().numpy())
                mod.register_forward_hook(hook_fn)

    def get(self):
        for k, v in self.acts.items():
            if not v:
                print(f"⚠️ No activations captured for layer: {k}")
        return {k: np.concatenate(v, axis=0) for k, v in self.acts.items() if v}


# Function to compute SVCCA similarity between two sets of layer activations
def svcca(a, b):
    out = {}
    for k in a:
        # Swap batch and feature dims: (B, C, H, W) → (C, B * H * W)
        A = a[k].transpose(1, 0, 2, 3).reshape(a[k].shape[1], -1)
        B = b[k].transpose(1, 0, 2, 3).reshape(b[k].shape[1], -1)
        out[k] = np.mean(robust_cca_similarity(A, B)['cca_coef1'])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_pre', type=str, required=True, help='Path to pretraind model config file')
    parser.add_argument('--config_rnd', type=str, required=True, help='Path to random weights model config file')
    parser.add_argument('--ckpt_pre', type=str, required=True, help='Checkpoint path of pretrained fine tuned model')
    parser.add_argument('--ckpt_rnd', type=str, required=True, action='store_true', help='Checkpoint path of random weights trained model')
    args = parser.parse_args()

    # ------- Pretrained SVCCA score ----------

    # Load model configuration
    params = Params(args.config_pre)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define layers to track SVCCA similarity on
    layers = ['model.features.conv0',
              'model.features.denseblock1.denselayer2.conv2',
              'model.features.denseblock1.denselayer6.conv2',
              'model.features.denseblock2.denselayer12.conv2',
              'model.features.denseblock3.denselayer48.conv2']

    # Load pretrained model (before fine-tuning)
    model_pre = DenseNet(params.dataset_name, pretrained=True).to(device).eval()

    # Load fine-tuned model (after training on ESC-50)
    # Initialize the model
    model_post = DenseNet(params.dataset_name, pretrained=True).to(device).eval()
    # Load the checkpoint
    checkpoint_path = args.ckpt_pre
    model_post = load_checkpoint_analysis(checkpoint_path, model_post)

    # Register hooks to capture activations
    h_pre = Hook(model_pre, layers)
    h_post = Hook(model_post, layers)

    # Load validation data (ESC-50 fold 1)
    val_loader = val_dataloader(params, fold_num=1)

    # Run one batch through both models to collect activations
    for x, _ in val_loader:
        x = x.to(device)
        with torch.no_grad():
            _ = model_pre(x)
            _ = model_post(x)
        break

    # Compute SVCCA scores between pretrained and fine-tuned model activations
    results = svcca(h_pre.get(), h_post.get())
    print("SVCCA Similarity Scores (Pretrained vs Fine-Tuned):")
    for layer, score in results.items():
        print(f"{layer}: {score:.4f}")

    # ------- Random weights SVCCA score ----------

    # Load model configuration
    params = Params(args.config_rnd)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained model (before fine-tuning)
    model_pre = DenseNet(params.dataset_name, pretrained=False).to(device).eval()

    # Load fine-tuned model (after training on ESC-50)
    # Initialize the model
    model_post = DenseNet(params.dataset_name, pretrained=False).to(device).eval()

    # Load the checkpoint
    checkpoint_path = args.ckpt_rnd
    model_post = load_checkpoint_analysis(checkpoint_path, model_post)

    # Register hooks to capture activations
    h_pre = Hook(model_pre, layers)
    h_post = Hook(model_post, layers)

    # Load validation data (ESC-50 fold 1)
    val_loader = val_dataloader(params, fold_num=1)

    # Run one batch through both models to collect activations
    for x, _ in val_loader:
        x = x.to(device)
        with torch.no_grad():
            _ = model_pre(x)
            _ = model_post(x)
        break

    results = svcca(h_pre.get(), h_post.get())
    print(results)

    # Print results
    print("SVCCA Similarity Scores (Random vs Trained):")
    for layer, score in results.items():
        print(f"{layer}: {score:.4f}")
