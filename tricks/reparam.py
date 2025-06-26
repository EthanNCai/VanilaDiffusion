import torch

# 设置环境
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化可导参数
mu = torch.tensor([0.0], requires_grad=True, device=device)
sigma = torch.tensor([1.0], requires_grad=True, device=device)

print("=== case1 ===")
x1 = torch.normal(mean=mu.detach(), std=sigma.detach()) 
x1 = x1.to(device)
loss1 = x1.mean()

try:
    loss1.backward() 
except RuntimeError as e:
    print("Error:", e)

print("x1.requires_grad:", x1.requires_grad)
print("x1.grad_fn:", x1.grad_fn)
print("mu.grad (non-reparam):", mu.grad)
print("sigma.grad (non-reparam):", sigma.grad)

mu.grad = None
sigma.grad = None

print("\n=== case2 ===")
epsilon = torch.randn_like(mu)   
x2 = mu + sigma * epsilon
loss2 = x2.mean()
loss2.backward()

print("x2.requires_grad:", x2.requires_grad)
print("x2.grad_fn:", x2.grad_fn)
print("mu.grad (reparam):", mu.grad.item())
print("sigma.grad (reparam):", sigma.grad.item())
