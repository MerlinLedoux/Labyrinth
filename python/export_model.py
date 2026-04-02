"""
Exporte les poids de l'acteur PPO vers public/model_weights.json.

Usage :
    python export_model.py [chemin_vers_modele]
    # par défaut : models/maze_ppo.pt

Format JSON (identique à l'ancien DQN — même inférence côté browser) :
{
  "obs_dim": 232,
  "action_dim": 4,
  "activation": "relu",
  "layers": [
    { "w": [[...]], "b": [...] },   // shared : 232 → 128
    { "w": [[...]], "b": [...] },   // shared : 128 → 128
    { "w": [[...]], "b": [...] },   // acteur : 128 → 4
  ]
}
"""
import sys
import os
import json
import torch
import torch.nn as nn
from train import ActorCritic

MODEL_PATH  = sys.argv[1] if len(sys.argv) > 1 else "models/maze_ppo.pt"
OUTPUT_PATH = "../public/model_weights.json"


def export(model_path: str, output_path: str) -> None:
    print(f"Chargement : {model_path}")
    model = ActorCritic()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    layers = []

    # Couches partagées (on ignore les ReLU)
    for module in model.shared:
        if isinstance(module, nn.Linear):
            layers.append({
                "w": module.weight.detach().numpy().tolist(),
                "b": module.bias.detach().numpy().tolist(),
            })

    # Tête acteur
    layers.append({
        "w": model.actor.weight.detach().numpy().tolist(),
        "b": model.actor.bias.detach().numpy().tolist(),
    })

    payload = {
        "obs_dim":    232,
        "action_dim": 4,
        "activation": "relu",
        "layers":     layers,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f)

    print(f"Exporté : {output_path}")
    for i, l in enumerate(layers):
        label = "shared" if i < 2 else "acteur"
        print(f"  couche {i} ({label}) : {len(l['w'][0])} → {len(l['w'])}")


if __name__ == "__main__":
    export(MODEL_PATH, OUTPUT_PATH)
