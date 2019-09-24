def sequential_search(arr, item):    # 顺序查找
    length = len(arr)
    for i in range(length):
        if arr[i] == item:
            return i
    return None


# 忽略了对数据类型，元素溢出等问题的判断。
class HashTable:    # 哈希查找
    def __init__(self, size):
        self.elem = [None for i in range(size)]  # 使用list数据结构作为哈希表元素保存方法
        self.count = size  # 最大表长

    def hash(self, key):
        return key % self.count  # 散列函数采用除留余数法

    def insert_hash(self, key):
        """插入关键字到哈希表内"""
        address = self.hash(key)  # 求散列地址
        while self.elem[address]:  # 当前位置已经有数据了，发生冲突。
            address = (address + 1) % self.count  # 线性探测下一地址是否可用
        self.elem[address] = key  # 没有冲突则直接保存。

    def search_hash(self, key):
        """查找关键字，返回布尔值"""
        star = address = self.hash(key)
        while self.elem[address] != key:
            address = (address + 1) % self.count
            if not self.elem[address] or address == star:  # 说明没找到或者循环到了开始的位置
                return False
        return True


if __name__ == '__main__':
    # 测试数组
    test = [10, 1, 15, 11, 3, -7, 6, 8, 12, 7, 14, -5, 9, 0, 13, 5, 1, -3, 4, -1, -2, -4, -10, 2]
    item = [10, 20]

    search_name = [
        'sequential'
    ]

    # Sequential_Search函数调用
    for j in range(len(search_name)):
        print("\n" + search_name[j] + "_search")
        for i in range(len(item)):
            result = eval(search_name[j] + '_search(test, item[i])')

            if result is not None:
                print("元素%d 在数组中的索引为 %d" % (item[i], result))
            else:
                print("元素%d 不在数组中" % item[i])

    print("\nHash_Search")
    hash_table = HashTable(len(test))
    for i in test:
        hash_table.insert_hash(i)
    # for i in hash_table.elem:
    #     if i:
    #         print((i, hash_table.elem.index(i)), end=" ")
    # print("\n")
    for i in range(len(item)):
        if hash_table.search_hash(item[i]):
            print("元素%d 在数组中" % item[i])
        else:
            print("元素%d 不在数组中" % item[i])
