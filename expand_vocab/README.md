
#### 扩张donut-base的词表
##### 增加换行符
- 1.在donut-tokenizer中"\n"会被处理成空格, 需额外增加特定的换行符号"</n>"

#### 扩充donut-base词表
- 1.使用expand_vocab.py扩充donut-base词表, 一次性操作

#### 扩充特定任务
- 1.针对特定任务增加特定所需要的词表, 每个任务由特定的字符
