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

在各诗正文末尾添加 * 开始符、# 结束符，按字训练词向量，构造 embed_mat

align() 将正文首字作为滑窗末字，填充或截取定长序列 align_seqs 和 next_inds

#### 5.build

train 80% / dev 20% 划分，next_inds 使用 to_categorical() 编码后超过内存限制

get_part() 每次读取 20% 的训练数据训练，通过单层和双层 rnn 构建语言生成模型

#### 6.generate

通过 word_inds 建立反向字典 ind_words，将输入序列按 seq_len 填充或截取

sample() 获取概率前 10 的字重新归一化，当逗号、句号为最大概率时直接返回

当长度小于 min_len 时生成结束符则丢弃并继续采样

当长度大于 min_len 时生成结束符或长度大于 max_len 时停止