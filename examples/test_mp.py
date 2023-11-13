import multiprocessing

"""
    多进程运行样例
"""
def worker(num, num2):
    """ 每个进程需要执行的任务 """
    print('Worker %d started' % num)
    return num * num2

if __name__ == '__main__':
    # 创建进程池并启动 4 个进程
    pool = multiprocessing.Pool(processes=4)
    # 创建一个进程共享队列
    result_queue = multiprocessing.Manager().Queue()
    # 5 次调用 worker 函数，将结果放入共享队列中
    for i in range(5):
        pool.apply_async(func=worker, args=(i,i), callback=result_queue.put)
    # 关闭进程池
    pool.close()
    # 等待所有进程结束
    pool.join()
    # 输出队列中的结果
    num_done = 0
    while num_done < 5:
        result = result_queue.get()
        print('Result:', result)
        num_done += 1
