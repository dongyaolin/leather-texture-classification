# get all the files of the path
import os
from PIL import Image


def get_Listfiles(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for file in files:
            # include path
            Filelist.append(os.path.join(home, file))
            # Filelist.append(file)

    return Filelist

src_path = "../row"
dis_path = "../row/new_class_1"
if not os.path.exists(dis_path):
    os.makedirs(dis_path)
file = get_Listfiles(src_path)
print(file)
c = 0
for img in file:
    try:
        im = Image.open(img)
        im.save(dis_path+'/'+ f'/{c}.jpg')
    except:
        pass
    c += 1
