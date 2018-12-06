import json

from random import shuffle


def check_index(quaples):
    nums = [fields[0] for fields in quaples]
    for i in range(len(nums) - 1):
        num1, num2 = [int(num) for num in nums[i].split('_')]
        next_num1, next_num2 = [int(num) for num in nums[i + 1].split('_')]
        if not (num1 == next_num1 and next_num2 - num2 == 1) and not \
               (next_num1 - num1 == 1 and next_num2 == 1):
            print('{}_{} -> {}_{}'.format(num1, num2, next_num1, next_num2))


def prepare(path_train_txt, path_train_csv, path_poetry, detail):
    quaples, poetry = list(), dict()
    with open(path_train_txt, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) != 4:
                continue
            quaples.append(fields)
            num, title, poet, text = fields
            if poet not in poetry:
                poetry[poet] = dict()
            if title not in poetry[poet]:
                poetry[poet][title] = list()
            poetry[poet][title].append(text)
    if detail:
        check_index(quaples)
    shuffle(quaples)
    with open(path_train_csv, 'w') as f:
        f.write('poet,title,text' + '\n')
        for num, title, poet, text in quaples:
            f.write(poet + ',' + title + ',' + text + '\n')
    with open(path_poetry, 'w') as f:
        json.dump(poetry, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    path_train_txt = 'data/train.txt'
    path_train_csv = 'data/train.csv'
    path_poetry = 'dict/poetry.json'
    prepare(path_train_txt, path_train_csv, path_poetry, detail=False)
