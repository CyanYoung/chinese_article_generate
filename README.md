## Chinese Poetry Generate 2018-8

#### 1.preprocess

prepare() 将 txt 数据处理为 (poet, title, text) 的三元组，保存为 csv 格式

建立 poetry 字典，实现作者、标题、正文的两层映射

check_index() 检查索引的连续性，对照全唐诗库找出缺损的数据

#### 2.retrieve

通过作者与部分标题，查找完整标题与正文

#### 3.explore

统计作者、标题、正文词汇、正文长度的频率，条形图可视化

计算 sent / word_per_sent 句词丰富度指标

#### 4.represent

在各诗正文末尾添加 # 结束符

#### 5.build



#### 6.generate

