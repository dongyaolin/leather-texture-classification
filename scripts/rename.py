"""该脚本重命名当前文件夹的文件名称，对重命名类别名称很与帮助
使用该脚本时将该脚本放在对应的文件夹下再执行脚本"""

import os

# 获取当前文件夹下所有文件夹的列表
folders = [name for name in os.listdir('.') if os.path.isdir(name)]

# 根据序号递增对文件夹进行重命名
for index, folder in enumerate(sorted(folders), start=1):
    new_name = f"class_{index}"  # 这里可以根据需要修改重命名规则
    os.rename(folder, new_name)
    print(f"Renaming {folder} to {new_name}")