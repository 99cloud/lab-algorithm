# Base

## Catalog

- [Sort](#Sort)
- [Search](#Search)

## Sort

详见【[PySort.py](https://github.com/99cloud/lab-algorithm/tree/master/Normal/Base/PySort.py)】

### 选择排序

选择排序是一种简单直观的排序算法，无论什么数据进去都是 O(n^2^) 的时间复杂度，所以用到它的时候，数据规模越小越好，唯一的好处可能就是不占用额外的内存空间了吧

- **算法步骤**
	- 首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置
	- 再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾
	- 重复第二步，直到所有元素均排序完毕

- **动图演示**

	<img src='img/selectionSort.gif' width=700/>

- **算法实现**

	```python
	def select_sort(arr):   # 选择排序
	    l = len(arr)
	
	    for j in range(0, l - 1):
	        count = j  # 记录最小元素下标
	        # 每次找出最小元素
	        for i in range(j, l - 1):
	            if arr[count] > arr[i + 1]:
	                count = i + 1
	        arr[j], arr[count] = arr[count], arr[j]     # 交换至前
	
	    return arr
	```

###  插入排序

插入排序是一种最简单直观的排序算法，它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入

- **算法步骤**

	- 将待排序序列第一个元素看做一个有序序列，把第二个元素到最后一个元素当成是未排序序列

	- 从头到尾依次扫描未排序序列，将扫描到的每个元素插入有序序列的适当位置（如果待插入的元素与有序序列中的某个元素相等，则将待插入元素插入到相等元素的后面）

- **算法实现**

	```python
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
	    return arr
	```

### 冒泡排序

冒泡排序是一种简单直观的排序算法，它重复地走访过要排序的数列，一次比较两个元素，如果他们的顺序错误就把他们交换过来，走访数列的工作是重复地进行直到没有再需要交换，也就是说该数列已经排序完成，这个算法的名字由来是因为越小的元素会经由交换慢慢“浮”到数列的顶端

- **算法步骤**
	- 比较相邻的元素。如果第一个比第二个大，就交换他们两个
	- 对每一对相邻元素作同样的工作，从开始第一对到结尾最后一对，这步做完后，最后的元素会是最大的数
	- 针对所有的元素重复以上的步骤，除了最后一个
	- 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较

- **动图演示**

	<img src='img/bubbleSort.gif' width=700/>

- **算法实现**

	```python
	def bubble_sort(arr):   # 冒泡排序
	    length = len(arr)
	    while length > 0:
	        for i in range(length - 1):
	            if arr[i] > arr[i + 1]:
	                swap(arr, i, i + 1)
	        length -= 1
	    return arr
	```

###  希尔排序

希尔排序，也称递减增量排序算法，是插入排序的一种更高效的改进版本，但希尔排序是非稳定排序算法

希尔排序是基于插入排序的以下两点性质而提出改进方法的：

- 插入排序在对几乎已经排好序的数据操作时，效率高，即可以达到线性排序的效率
- 但插入排序一般来说是低效的，因为插入排序每次只能将数据移动一位

希尔排序的基本思想是：先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序，待整个序列中的记录“基本有序”时，再对全体记录进行依次直接插入排序。

- **算法步骤**
	- 选择一个增量序列 t~1~，t~2~，……，t~k~，其中 t~i~ > t~j~, t~k~ = 1
	- 按增量序列个数 k，对序列进行 k 趟排序
	- 每趟排序，根据对应的增量 t~i~，将待排序列分割成若干长度为 m 的子序列，分别对各子表进行直接插入排序,仅增量因子为 1 时，整个序列作为一个表来处理，表长度即为整个序列的长度

- **Python 代码实现**

	```python
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
	    return arr
	```

### 快速排序

快速排序是由东尼·霍尔所发展的一种排序算法，在平均状况下，排序 n 个项目要 Ο(nlogn) 次比较，在最坏状况下则需要 Ο(n^2^) 次比较，但这种状况并不常见，事实上，快速排序通常明显比其他 Ο(nlogn) 算法更快，因为它的内部循环（inner loop）可以在大部分的架构上很有效率地被实现出来。

快速排序使用分治法（Divide and conquer）策略来把一个串行（list）分为两个子串行（sub-lists）

快速排序又是一种分而治之思想在排序算法上的典型应用，本质上来看，快速排序应该算是在冒泡排序基础上的递归分治法。

快速排序的名字起的是简单粗暴，因为一听到这个名字你就知道它存在的意义，就是快，而且效率高！它是处理大数据最快的排序算法之一了。虽然 Worst Case 的时间复杂度达到了 O(n²)，但在大多数情况下都比平均时间复杂度为 O(n logn) 的排序算法表现要更好

> 快速排序的最坏运行情况是 O(n²)，比如说顺序数列的快排。但它的平摊期望时间是 O(nlogn)，且 O(nlogn) 记号中隐含的常数因子很小，比复杂度稳定等于 O(nlogn) 的归并排序要小很多，所以，对绝大多数顺序性较弱的随机数列而言，快速排序总是优于归并排序

- **算法步骤**
	- 从数列中挑出一个元素，称为 “基准”（pivot）
	- 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边），在这个分区退出之后，该基准就处于数列的中间位置，这个称为分区（partition）操作
	- 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序

	递归的最底部情形，是数列的大小是零或一，也就是永远都已经被排序好了。虽然一直递归下去，但是这个算法总会退出，因为在每次的迭代（iteration）中，它至少会把一个元素摆到它最后的位置去

- **算法实现**

	```python
	def quick_sort(arr):   # 快速排序
	    if len(arr) < 2:
	        return arr
	    else:
	        pivot = arr[0]
	        less = [i for i in arr[1:] if i <= pivot]
	        greater = [i for i in arr[1:] if i > pivot]
	    return quick_sort(less) + [pivot] + quick_sort(greater)
	```

### 归并排序

归并排序（Merge sort）是建立在归并操作上的一种有效的排序算法，该算法是采用分治法（Divide and Conquer）的一个非常典型的应用

作为一种典型的分而治之思想的算法应用，归并排序的实现由两种方法

- 自上而下的递归（所有递归的方法都可以用迭代重写，所以就有了第 2 种方法）
- 自下而上的迭代

和选择排序一样，归并排序的性能不受输入数据的影响，但表现比选择排序好的多，因为始终都是 O(nlogn) 的时间复杂度，代价是需要额外的内存空间

- **算法步骤**
	- 申请空间，使其大小为两个已经排序序列之和，该空间用来存放合并后的序列
	- 设定两个指针，最初位置分别为两个已经排序序列的起始位置
	- 比较两个指针所指向的元素，选择相对小的元素放入到合并空间，并移动指针到下一位置
	- 重复步骤 3 直到某一指针达到序列尾
	- 将另一序列剩下的所有元素直接复制到合并序列尾

- 算法实现

	```python
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
	```

## Search

### 有序序列搜索

详见【[PyOrderSearch.py](https://github.com/99cloud/lab-algorithm/tree/master/Normal/Base/PyOrderSearch.py)】

#### 二分查找

二分查找，是一种在有序数组中查找某一特定元素的查找算法，查找过程从数组的中间元素开始，如果中间元素正好是要查找的元素，则查找过程结束，如果某一特定元素大于或者小于中间元素，则在数组大于或小于中间元素的那一半中查找，而且跟开始一样从中间元素开始比较，如果在某一步骤数组为空，则代表找不到，这种查找算法每一次比较都使查找范围缩小一半

- 算法描述

	给予一个包含 n 个带值元素的数组 A
	- 令 L 为 0 ， R 为 n-1
	- 如果 L>R ，则搜索以失败告终 
	- 令 m （中间值元素）为  [(L+R)/2]
	- 如果 A~m~<T ，令 L 为 m + 1 并回到步骤 2
	- 如果 A~m~>T，令 R 为 m - 1 并回到步骤 2

- 复杂度分析

	时间复杂度：折半搜索每次把搜索区域减少一半，时间复杂度为 **O(logn)**    空间复杂度为 **O(1)**

- 算法实现

	```python
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
	```

#### 插值查找

插值查找是根据要查找的关键字 key 与查找表中最大最小记录的关键字比较后的查找方法，其核心就在于插值的计算公式 (key-a[low])/(a[high]-a[low])*(high-low)
时间复杂度o(logn)但对于表长较大而关键字分布比较均匀的查找表来说，效率较高

- 算法思想
	基于二分查找算法，将查找点的选择改进为自适应选择，可以提高查找效率。当然，差值查找也属于有序查找
	注：对于表长较大，而关键字分布又比较均匀的查找表来说，插值查找算法的平均性能比折半查找要好的多，反之，数组中如果分布非常不均匀，那么插值查找未必是很合适的选择
- 复杂度分析
	- 时间复杂性：如果元素均匀分布，则O(log n)，在最坏的情况下可能需要 O(n)
	- 空间复杂度：O(1)

- 算法实现

	```python
	def interpolate_search(arr, item):  # 插值查找
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
	```

### 无序序列搜索

详见【[PySearch.py](https://github.com/99cloud/lab-algorithm/tree/master/Normal/Base/PySearch.py)】

#### 顺序查找

顺序查找又称为线性查找，是一种最简单的查找方法，适用于线性表的顺序存储结构和链式存储结构，该算法的时间复杂度为 O(n)

- 基本思路

	从第一个元素 m 开始逐个与需要查找的元素 x 进行比较，当比较到元素值相同（即 m=x ）时返回元素 m 的下标，如果比较到最后都没有找到，则返回 -1

- 优缺点
	- 缺点：是当 n 很大时，平均查找长度较大，效率低
	- 优点：是对表中数据元素的存储没有要求，另外，对于线性链表，只能进行顺序查找

- 算法实现

	```python
	def sequential_search(arr, item):    # 顺序查找
	    length = len(arr)
	    for i in range(length):
	        if arr[i] == item:
	            return i
	    return None
	```

####哈希查找

哈希表是一种以键-值（key-indexed）存储数据的结构，只要输入待查找的值即 key，即可查找到其对应的值

- 算法思想
	哈希的思路很简单，如果所有的键都是整数，那么就可以使用一个简单的无序数组来实现，将键作为索引，值即为其对应的值，这样就可以快速访问任意键的值，这是对于简单的键的情况，我们将其扩展到可以处理更加复杂的类型的键

- 算法流程
	- 用给定的哈希函数构造哈希表
	- 根据选择的冲突处理方法解决地址冲突
		常见的解决冲突的方法：拉链法和线性探测法
	- 在哈希表的基础上执行哈希查找

- 复杂度分析
	单纯论查找复杂度：对于无冲突的 Hash 表而言，查找复杂度为 O(1)（注意，在查找之前我们需要构建相应的Hash表）

- 算法实现

	```python
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
	```

### 二叉树

详见【[PyBinaryTree.py](https://github.com/99cloud/lab-algorithm/tree/master/Normal/Base/PyBinaryTree.py)】

二叉查找树是先对待查找的数据进行生成树，确保树的左分支的值小于右分支的值，然后在就行和每个节点的父节点比较大小，查找最适合的范围，这个算法的查找效率很高，但是如果使用这种查找方法要首先创建树

- 算法思想

	二叉查找树（BinarySearch Tree）或者是一棵空树，或者是具有下列性质的二叉树
	- 若任意节点的左子树不空，则左子树上所有结点的值均小于它的根结点的值
	- 若任意节点的右子树不空，则右子树上所有结点的值均大于它的根结点的值
	- 任意节点的左、右子树也分别为二叉查找树

- 复杂度分析 

	它和二分查找一样，插入和查找的时间复杂度均为 O(logn) ，但是在最坏的情况下仍然会有 O(n) 的时间复杂度，原因在于插入和删除元素的时候，树没有保持平衡

- 算法实现

	```python
	class BSTNode:  # 定义一个二叉树节点类
	    def __init__(self, data, left=None, right=None):
	        """
	        初始化
	        :param data: 节点储存的数据
	        :param left: 节点左子树
	        :param right: 节点右子树
	        """
	        self.data = data
	        self.left = left
	        self.right = right
	
	class BinarySortTree:   # 基于BSTNode类的二叉查找树，维护一个根节点的指针
	    def __init__(self):
	        self._root = None
	    def is_empty(self):
	        return self._root is None
	    def search(self, key):
	        """
	        关键码检索
	        :param key: 关键码
	        :return: 查询节点或None
	        """
	        bt = self._root
	        while bt:
	            entry = bt.data
	            if key < entry:
	                bt = bt.left
	            elif key > entry:
	                bt = bt.right
	            else:
	                return entry
	        return None
	    def insert(self, key):
	        """
	        插入操作
	        :param key:关键码
	        :return: 布尔值
	        """
	        bt = self._root
	        if not bt:
	            self._root = BSTNode(key)
	            return
	        while True:
	            entry = bt.data
	            if key < entry:
	                if bt.left is None:
	                    bt.left = BSTNode(key)
	                    return
	                bt = bt.left
	            elif key > entry:
	                if bt.right is None:
	                    bt.right = BSTNode(key)
	                    return
	                bt = bt.right
	            else:
	                bt.data = key
	                return
	    def delete(self, key):
	        """
	        二叉查找树最复杂的方法
	        :param key: 关键码
	        :return: 布尔值
	        """
	        p, q = None, self._root     # 维持p为q的父节点，用于后面的链接操作
	        if not q:
	            print("空树！")
	            return
	        while q and q.data != key:
	            p = q
	            if key < q.data:
	                q = q.left
	            else:
	                q = q.right
	            if not q:               # 当树中没有关键码key时，结束退出。
	                return
	        # 上面已将找到了要删除的节点，用q引用。而p则是q的父节点或者None（q为根节点时）。
	        if not q.left:
	            if p is None:
	                self._root = q.right
	            elif q is p.left:
	                p.left = q.right
	            else:
	                p.right = q.right
	            return
	        # 查找节点q的左子树的最右节点，将q的右子树链接为该节点的右子树
	        # 该方法可能会增大树的深度，效率并不算高。可以设计其它的方法。
	        r = q.left
	        while r.right:
	            r = r.right
	        r.right = q.right
	        if p is None:
	            self._root = q.left
	        elif p.left is q:
	            p.left = q.left
	        else:
	            p.right = q.left
	    def __iter__(self):
	        """
	        实现二叉树的中序遍历算法,
	        展示我们创建的二叉查找树.
	        直接使用python内置的列表作为一个栈。
	        :return: data
	        """
	        stack = []
	        node = self._root
	        while node or stack:
	            while node:
	                stack.append(node)
	                node = node.left
	            node = stack.pop()
	            yield node.data
	            node = node.right
	```

	【[返回顶部](#Base)】