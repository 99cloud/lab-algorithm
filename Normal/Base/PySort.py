def select_sort(arr):   # 选择排序
    l = len(arr)

    for j in range(0, l - 1):
        count = j  # 记录最小元素下标
        # 每次找出最小元素
        for i in range(j, l - 1):
            if arr[count] > arr[i + 1]:
                count = i + 1
        arr[j], arr[count] = arr[count], arr[j]  # 实现跟上述代码一样

    print('Sorted List:', arr)


def insert_sort(arr):   # 插入排序
    l = len(arr)
    for i in range(1, l):
        temp = arr[i]
        for j in range(i - 1, -1, -1):
            if temp < arr[j]:  # 如果第i个元素小于前i个元素中的第j个
                arr[j + 1] = arr[j]  # 则第j个元素先后移1位
                arr[j] = temp  # 将i个元素赋值给空着的位置
            else:  # 如果第i个元素大于等于前i个元素中的第j个则结束循环
                break

    print('Sorted List:', arr)


# def shell_sort(arr):    # 希尔排序
#     l = int(len(arr) / 2)
#     # 生成增量列表
#     gap_list = []
#     while l > 0:
#         gap_list.append(l)
#         l = int(l / 2)
#     print('gap_list: ', gap_list)
#
#     for gap in gap_list:  # 增量gap，并逐步缩小增量
#         # print(gap)
#         for i in range(gap, len(arr)):  # 从第gap个元素，逐个对其所在组进行直接插入排序操作
#             j = i
#             while j - gap >= 0 and arr[j - gap] > arr[j]:
#                 swap(arr, j, j - gap)  # 交换两个元素
#                 j = j - gap
#
#     print('Sorted List:', arr)


def shell_sort(arr):    # 希尔排序
    N = len(arr)
    h = 1
    gap_list = [h]

    while h < N/3:
        h = 3 * h + 1   # 1, 4, 13, 40, 121, 364, 1093, ...
        gap_list.append(h)
    print('gap_list: ', gap_list)

    while h >= 1:
        # 将数组变为h有序
        for i in range(h, N):
            # 将arr[i]插入到arr[i - h], arr[i - 2 * h], ... 之中
            j = i
            while j >= h and arr[j] < arr[j - h]:
                swap(arr, j, j - h)
                j -= h
        h = int(h / 3)

    print('Sorted List:', arr)


def swap(arr, a, b):    # 交换arr[a]和arr[b]
    arr[a] = arr[a] + arr[b]
    arr[b] = arr[a] - arr[b]
    arr[a] = arr[a] - arr[b]


def bubble_sort(arr):   # 冒泡排序
    length = len(arr)
    while length > 0:
        for i in range(length - 1):
            if arr[i] > arr[i + 1]:
                swap(arr, i, i + 1)
        length -= 1

    print('Sorted List:', arr)


def quick_sort(arr):   # 快速排序
    if len(arr) < 2:
        return arr
    else:
        pivot = arr[0]
        less = [i for i in arr[1:] if i <= pivot]
        greater = [i for i in arr[1:] if i > pivot]
    return quick_sort(less) + [pivot] + quick_sort(greater)


def merge_sort(lists):  # 归并排序
    if len(lists) <= 1:
        return lists
    num = int(len(lists)/2)
    left = merge_sort(lists[:num])  # 将列表从中间分为两部分
    right = merge_sort(lists[num:])
    return merge(left, right)       # 合并两个列表


def merge(left, right):
    r, l = 0, 0
    result = []
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += left[l:]
    result += right[r:]
    return result


if __name__ == "__main__":
    test = [10, 1, 15, 11, 3, -7, 6, 8, 12, 7, 14, -5, 9, 0, 13, 5, 1, -3, 4, -1, -2, -4, -10, 2]

    print("\nSelect_Sort")
    select_sort(test)

    print("\nInsert_Sort")
    insert_sort(test)

    print("\nShell_Sort")
    shell_sort(test)

    print("\nBubble_Sort")
    bubble_sort(test)

    print("\nQuick_Sort")
    quick_list = quick_sort(test)
    print('Sorted List:', quick_list)

    print("\nMerge_Sort")
    merge_list = merge_sort(test)
    print('Sorted List:', merge_list)

