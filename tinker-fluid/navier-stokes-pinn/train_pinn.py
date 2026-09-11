import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Parameters
nu = 0.01  # viscosity
N_coll = 10000  # Collocation points (physics)
N_ic = 1000  # Initial condition points

# Analytical solution (benchmark)
def taylor_green_u(x, y, t):
    return -np.cos(x) * np.sin(y) * np.exp(-2 * nu * t)

def taylor_green_v(x, y, t):
    return np.sin(x) * np.cos(y) * np.exp(-2 * nu * t)

def taylor_green_p(x, y, t):
    return -0.25 * (np.cos(2 * x) + np.cos(2 * y)) * np.exp(-4 * nu * t)

class PINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), # Input: x, y, t
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)  # Output: u, v, p
        )

    def forward(self, xyt):
        return self.net(xyt) # (N, 3) -> (N, 3)

def get_training_data(t_max=5):
    # Collocation points: x, y in [0, 2pi], t in [0, t_max]
    x = np.random.uniform(0, 2*np.pi, (N_coll, 1))
    y = np.random.uniform(0, 2*np.pi, (N_coll, 1))
    t = np.random.uniform(0, t_max, (N_coll, 1))
    xyt_coll = torch.tensor(np.hstack([x, y, t]), dtype=torch.float32, requires_grad=True).to(device)

    # Normal data (Initial condition: t = 0)
    x = np.random.uniform(0, 2*np.pi, (N_ic, 1))
    y = np.random.uniform(0, 2*np.pi, (N_ic, 1))
    t = np.zeros((N_ic, 1))
    u = taylor_green_u(x, y, t)
    v = taylor_green_v(x, y, t)
    xyt_ic = torch.tensor(np.hstack([x, y, t]), dtype=torch.float32).to(device)
    uv_ic = torch.tensor(np.hstack([u, v]), dtype=torch.float32).to(device)

    return xyt_coll, xyt_ic, uv_ic

# Physics loss
def compute_physics_loss(model, xyt_coll):
    xyt_coll = xyt_coll.clone().detach().requires_grad_(True)

    uvp = model(xyt_coll)
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    grads = torch.autograd.grad(u, xyt_coll, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    grads = torch.autograd.grad(v, xyt_coll, torch.ones_like(v), create_graph=True)[0]
    v_x = grads[:, 0:1]
    v_y = grads[:, 1:2]
    v_t = grads[:, 2:3]

    grads = torch.autograd.grad(p, xyt_coll, torch.ones_like(p), create_graph=True)[0]
    p_x = grads[:, 0:1]
    p_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, xyt_coll, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyt_coll, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    v_xx = torch.autograd.grad(v_x, xyt_coll, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, xyt_coll, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

    continuity = u_x + v_y

    mom_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    loss_pde = (mom_x**2).mean() + (mom_y**2).mean() + (continuity**2).mean()
    return loss_pde

# Initial condition loss
def compute_ic_loss(model, xyt_ic, uv_ic):
    pred_uv = model(xyt_ic)[:, :2]
    return ((pred_uv - uv_ic) ** 2).mean()

# === Training ===
xyt_coll, xyt_ic, uv_ic = get_training_data()

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5000
loss_history = []
print("Starting training...")
for epoch in range(epochs):
    optimizer.zero_grad()
    loss_pde = compute_physics_loss(model, xyt_coll)
    loss_ic = compute_ic_loss(model, xyt_ic, uv_ic)
    loss = loss_pde + loss_ic
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0 or epoch == epochs - 1:
        l = loss.item()
        pde_l = loss_pde.item()
        ic_l = loss_ic.item()
        print(f"Epoch {epoch:04d} | Loss: {l:.5e} | PDE: {pde_l:.5e} | IC: {ic_l:.5e}")
        loss_history.append({"epoch": epoch, "loss": l, "pde_loss": pde_l, "ic_loss": ic_l})

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model_weights.pth")

# === Evaluation ===
model.eval()

def plot_and_evaluate(model, t_val=0.5, suffix=""):
    N = 50
    x = np.linspace(0, 2*np.pi, N)
    y = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t_val)

    xyt = torch.tensor(np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1), dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(xyt).cpu().numpy()
    
    U_pred = pred[:, 0].reshape(N, N)
    V_pred = pred[:, 1].reshape(N, N)
    P_pred = pred[:, 2].reshape(N, N)

    U_true = taylor_green_u(X, Y, T)
    V_true = taylor_green_v(X, Y, T)
    P_true = taylor_green_p(X, Y, T)

    # For pressure, PINN predicts pressure up to a constant since only grad(p) is in N-S.
    # We mean-center both pressures before computing error.
    P_pred_centered = P_pred - np.mean(P_pred)
    P_true_centered = P_true - np.mean(P_true)

    # Metrics
    rmse_u = np.sqrt(np.mean((U_pred - U_true)**2))
    rmse_v = np.sqrt(np.mean((V_pred - V_true)**2))
    rmse_p = np.sqrt(np.mean((P_pred_centered - P_true_centered)**2))
    
    rel_u = np.linalg.norm(U_pred - U_true) / np.linalg.norm(U_true)
    rel_v = np.linalg.norm(V_pred - V_true) / np.linalg.norm(V_true)
    rel_p = np.linalg.norm(P_pred_centered - P_true_centered) / np.linalg.norm(P_true_centered)

    # Velocity Comparison Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Taylor-Green Vortex at t = {t_val} {suffix}", fontsize=16)
    
    axs[0, 0].quiver(X, Y, U_true, V_true, scale=20)
    axs[0, 0].set_title("Analytical Velocity Field")

    axs[0, 1].quiver(X, Y, U_pred, V_pred, scale=20)
    axs[0, 1].set_title("PINN Predicted Velocity Field")

    diff_u = np.abs(U_pred - U_true)
    diff_v = np.abs(V_pred - V_true)

    im1 = axs[1, 0].imshow(diff_u, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower')
    axs[1, 0].set_title("Absolute Error in u")
    fig.colorbar(im1, ax=axs[1, 0])

    im2 = axs[1, 1].imshow(diff_v, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower')
    axs[1, 1].set_title("Absolute Error in v")
    fig.colorbar(im2, ax=axs[1, 1])

    plt.tight_layout()
    plot_file = f"eval_t{t_val}_velocity{suffix}.png"
    plt.savefig(plot_file)
    plt.close()

    # Pressure Comparison Plot
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle(f"Pressure Field at t = {t_val} (Mean-centered)", fontsize=16)
    
    im_p1 = axs2[0].imshow(P_true_centered, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower', cmap='coolwarm')
    axs2[0].set_title("Analytical Pressure")
    fig2.colorbar(im_p1, ax=axs2[0])

    im_p2 = axs2[1].imshow(P_pred_centered, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower', cmap='coolwarm')
    axs2[1].set_title("PINN Predicted Pressure")
    fig2.colorbar(im_p2, ax=axs2[1])

    diff_p = np.abs(P_pred_centered - P_true_centered)
    im_p3 = axs2[2].imshow(diff_p, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower')
    axs2[2].set_title("Absolute Error in Pressure")
    fig2.colorbar(im_p3, ax=axs2[2])
    
    plt.tight_layout()
    plot_file2 = f"eval_t{t_val}_pressure{suffix}.png"
    plt.savefig(plot_file2)
    plt.close()

    metrics = {
        "t": t_val,
        "rmse_u": rmse_u,
        "rmse_v": rmse_v,
        "rmse_p": rmse_p,
        "rel_u": rel_u,
        "rel_v": rel_v,
        "rel_p": rel_p
    }
    return metrics, plot_file, plot_file2

results = []
for t in [0.0, 0.5, 2.0]:
    metrics, pf1, pf2 = plot_and_evaluate(model, t_val=t)
    results.append({
        "metrics": metrics,
        "plot_velocity": pf1,
        "plot_pressure": pf2
    })

with open("training_results.json", "w") as f:
    json.dump({
        "loss_history": loss_history,
        "eval_results": results
    }, f, indent=2)

print("Evaluation complete. Results saved.")
