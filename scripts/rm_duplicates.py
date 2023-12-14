"""该脚本对重复的图片进行去重，通常用于对原始数据的重复图片进行筛除"""

from hashlib import md5
import os
import time


def get_md5(filename):
    file_txt = open(filename, 'rb').read()
    m = md5(file_txt)
    return m.hexdigest()


def main():
    allfiles = r"../row"
    # all_md5 = []
    # total_file = 0
    # total_delete = 0
    start = time.time()
    for dir in os.listdir(allfiles):
        all_md5 = []
        total_file = 0
        total_delete = 0
        path = os.path.join(allfiles, dir)
        print(path)
        for file in os.listdir(path):
            total_file += 1;
            real_path = os.path.join(path, file)
            if os.path.isfile(real_path) == True:
                filemd5 = get_md5(real_path)
                if filemd5 in all_md5:
                    total_delete += 1
                    os.remove(real_path)
                    # print u'删除', file
                else:
                    all_md5.append(filemd5)
        end = time.time()
        time_last = end - start
        print(
            u'文件总数：', total_file)
        print(
            u'删除个数：', total_delete)
    print(
        u'耗时：', time_last, u'秒')


if __name__ == '__main__':
    main()
