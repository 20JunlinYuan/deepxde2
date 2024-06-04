import torch

# 假设你有一个形状为[35, 1]的张量 tensor
tensor = torch.randn(35, 1)

# 使用逻辑运算符判断张量中的每个值是否大于0
condition = tensor > 0
print("condition: ", condition)
# 使用逻辑索引选择满足条件的元素，并保持形状一致
positive_values = torch.zeros_like(tensor)
positive_values[condition] = tensor[condition]

# 对满足条件的元素进行进一步的运算，例如加倍
result = positive_values * 2

print("满足条件的正值元素：", positive_values)
print("进一步运算后的结果：", result)