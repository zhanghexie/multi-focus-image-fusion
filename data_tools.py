"""
函数：
    upickle：加载数据
    savedata：保存数据
"""
import pickle

def unpickle(path):
    """
    加载pickle数据
    参数：
        file_path：pickle文件路径
    返回值：
        读取的数据
    """
    with open(path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def save_data(data,path):
    """
    参数：
        file_path: 文件路径
        data : 要写入的文件
    """
    with open(path, 'wb') as fo:
        pickle.dump(data,fo)
    return
