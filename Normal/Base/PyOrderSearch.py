# 返回 item 在 arr 中的索引，如果不存在返回 None


def binarysearch(arr, l, r, item):  # 二分查找（递归）
    # 基本判断
    if r >= l:
        mid = int(l + (r - l) / 2)
        # 元素整好的中间位置
        if arr[mid] == item:
            return mid
        # 元素小于中间位置的元素，只需要再比较左边的元素
        elif arr[mid] > item:
            return binarysearch(arr, l, mid - 1, item)
        # 元素大于中间位置的元素，只需要再比较右边的元素
        else:
            return binarysearch(arr, mid + 1, r, item)
    else:
        # 不存在
        return None


def binary_search(arr, item):       # 二分查找
    low = 0
    high = len(arr) - 1
    count = 0
    while low <= high:
        count += 1
        mid = int((low + high) / 2)
        guess = arr[mid]
        if guess == item:
            return mid, count
        elif guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None


def interpolate_search(arr, item):
    low = 0
    high = len(arr) - 1
    count = 0
    while low < high:
        count += 1
        # 计算mid值是插值算法的核心代码
        mid = low + int((high - low) * (item - arr[low])/(arr[high] - arr[low]))
        # print("mid=%s, low=%s, high=%s" % (mid, low, high))
        if item < arr[mid]:
            high = mid - 1
        elif item > arr[mid]:
            low = mid + 1
        else:
            return mid, count
    return None


if __name__ == '__main__':
    # 测试数组
    arr = [-10, -8, -5, -2, 0, 1, 4, 7, 10, 17, 21, 25, 28, 35, 55, 63, 78, 99,
           118, 188, 219, 314, 369, 442, 520, 618, 945, 1991]
    item = [-10, 10, 618, 20]

    # binarysearch函数调用
    print("\nbinarysearch")
    for i in range(len(item)):
        result = binarysearch(arr, 0, len(arr) - 1, item[i])

        if result is not None:
            print("元素% 4d 在数组中的索引为% 3d" % (item[i], result))
        else:
            print("元素% 4d 不在数组中" % item[i])
    
    # binary_search函数调用
    print("\nbinary_search")
    for i in range(len(item)):
        result = binary_search(arr, item[i])

        if result is not None:
            print("元素% 4d 在数组中的索引为% 3d, 搜索次数为% 2d" % (item[i], result[0], result[1]))
        else:
            print("元素% 4d 不在数组中" % item[i])

    # interpolate_search函数调用
    print("\ninterpolate_search")
    for i in range(len(item)):
        result = interpolate_search(arr, item[i])

        if result is not None:
            print("元素% 4d 在数组中的索引为% 3d, 搜索次数为% 2d" % (item[i], result[0], result[1]))
        else:
            print("元素% 4d 不在数组中" % item[i])
