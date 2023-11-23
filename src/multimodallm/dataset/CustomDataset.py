
"""
    在训练过程中：RuntimeError: [enforce fail at C:\cb\pytorch_1000000000000\work\c10\core\impl\alloc_cpu.cpp:81] data. DefaultCPUAllocator: not enough memory: you tried to allocate 1048576 bytes. Error in sys.excepthook:
    初步怀疑是不是把en_ds全部加入到内存导致的？
    需要定义一个dataset, 每次需要使用市在加载
"""