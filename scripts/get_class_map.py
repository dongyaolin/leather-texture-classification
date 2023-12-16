import os

# 获取当前目录下的所有文件名
files = [f for f in os.listdir('.') if os.path.isdir(f)]

# 构建索引字典
index_dict = {index: filename for index, filename in enumerate(files)}

# 打印索引字典
print(index_dict)
