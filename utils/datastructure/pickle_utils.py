import base64
import pickle


def serialize(obj, binary=True):
    # 序列化。将数据对象序列化为字节对象
    pickled_string = pickle.dumps(obj)
    if binary:  # 将序列化后的字节对象进一步转换为二进制字符串
        pickled_string = base64.b64encode(pickled_string)
    return pickled_string


def deserialize(pickled_string, binary=True):
    # 将反序列化。将字节对象反序列化为数据对象
    if binary:  # 将二进制字符串转换为pickle可识别的字节对象
        pickled_string = base64.b64decode(pickled_string)
    obj = pickle.loads(pickled_string)
    return obj


if __name__ == '__main__':
    data = [1, 2, 3]
    data_bin_string = serialize(data)
    print('binary serialized:', data_bin_string)
    re_data = deserialize(data_bin_string)
    print('binary deserialized:', re_data)

    data_string = serialize(data, binary=False)
    print('common serialized:', data_string)
    re_data = deserialize(data_string, binary=False)
    print('common deserialized:', re_data)
