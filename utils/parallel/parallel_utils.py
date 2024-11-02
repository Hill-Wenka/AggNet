import concurrent.futures
import multiprocessing as mp

# from tqdm import tqdm
from tqdm.notebook import tqdm

'''
zip_params = list(zip(a, b)): [(a1, b1), (a2, b2), ...]
unzip_params = list(zip(*zip_params)): [(a1, a2, ...), (b1, b2, ...)]
调用所有异步并行函数时，统一使用zip_params的形式传入参数params。
对于concurrent_map，其函数内会将zip_params转换为unzip_params作为executor.map()的参数，所以调用时直接传入zip_params即可。
'''


def async_parallel(parallel_func, params, cpu_num=mp.cpu_count(), desc=None, kwds=None):
    # 异步并行
    # params: list of tuple, e.g. [(a1, b1), (a2, b2), ...]
    kwds = {} if kwds is None else kwds
    pbar = tqdm(total=len(params))
    desc = desc if desc is not None else 'Parallel Running, cpu_num: %d' % cpu_num
    pbar.set_description(desc)
    update = lambda *args: pbar.update()

    pool = mp.Pool(cpu_num)
    processes = [pool.apply_async(parallel_func, args=param, kwds=kwds, callback=update) for param in params]
    results = []
    for p in processes:
        try:
            results.append(p.get())
        except Exception as e:
            print('Exception: ', e)
            results.append(None)
    pool.close()
    pool.join()
    return results


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


def concurrent_map(parallel_func, params, cpu_num=mp.cpu_count()):
    # 异步并行，但是保证results的顺序按照params的顺序而不是执行顺序。缺点是如果有个别任务有异常会阻塞
    # params: list of tuple, e.g. [(a1, b1), (a2, b2), ...]
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num) as executor:
        results = executor.map(parallel_func, *zip(*params))  # *zip(*params) 将参数转换为 [(a1, a2, ...), (b1, b2, ...)]
    return list(results)
