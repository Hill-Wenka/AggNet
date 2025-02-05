import concurrent.futures
import multiprocessing as mp

from tqdm.notebook import tqdm

'''
zip_params = list(zip(a, b)): [(a1, b1), (a2, b2), ...]
unzip_params = list(zip(*zip_params)): [(a1, a2, ...), (b1, b2, ...)]
调用所有异步并行函数时，统一使用zip_params的形式传入参数params。
对于concurrent_map，其函数内会将zip_params转换为unzip_params作为executor.map()的参数，所以调用时直接传入zip_params即可。
'''


def concurrent_submit(parallel_func, params, cpu_num=mp.cpu_count(), desc=None):
    # 异步并行，但是results的顺序是随机的。优点是即使个别任务有异常也不会阻塞
    # params: list of tuple, e.g. [(a1, b1), (a2, b2), ...]
    pbar = tqdm(total=len(params))
    desc = desc if desc is not None else 'Parallel Running, cpu_num: %d' % cpu_num
    pbar.set_description(desc)

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num) as executor:
        futures = [executor.submit(parallel_func, *param) for param in params]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print('concurrent submit Exception: ', e)
                results.append(None)
            pbar.update(1)
    return results
