def _nest_dict_rec(k, v, out):
    # 递归创建嵌套字典
    k, *rest = k.split('.', 1)
    if rest:
        _nest_dict_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v


def nest_dict(flat):
    # 将字典flat中包含"."的key进行拆分，创建嵌套字典
    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result)
    return result


def merge_dicts(dict_list):
    # 合并字典，如果有重复的key，后面的会覆盖前面的
    result = {}
    for d in dict_list:
        result.update(d)
    return result
