## Chinese Poetry Generate 2018-8

#### 1.preprocess

prepare() 将 txt 数据处理为 (poet, title, text) 的三元组，保存为 csv 格式

建立 poetry 字典，实现作者、标题、正文的两层映射

check_index() 检查索引的连续性，对照全唐诗库找出缺损的数据

#### 2.retrieve

输入作者与关键词，查找所有包含关键词的标题及正文

#### 3.explore

统计作者、标题、正文词汇、正文长度的频率，条形图可视化

#### 4.represent

add_flag() 添加控制符，word2vec() 按字训练词向量、构造 embed_mat

shift() 分别删去 bos、eos 得到 sent、label，align() 分别截取或填充为定长序列

#### 5.build

label 先进行 expand_dims 再使用 sparse_categorical_crossentropy 

避免 to_categorical() 超过内存限制， 通过 rnn、cnn 构建语言生成模型

#### 6.generate

通过 word_inds 建立反向字典 ind_words，截取或填充输入序列

对概率前 5 的输出归一化后采样，逗号、句号为最大概率则直接返回

生成结束符或长度大于 max_len 时停止、小于 min_len 则采样直到非结束符