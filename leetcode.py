#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from boxx import *
#import yllab as yl
from dataStruct import *
from copy import deepcopy
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

        
        

class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(nums)
        l,r=0,n-1
        while 1:
            m = (l+r)//2
            if r<l:
                return [-1, -1]
            if nums[m] == target:
                ind = m
                break
            if nums[m] >= target:
                l,r = l,m-1
            else:
                l,r = m+1,r
        l,r = ind, ind
        while l >= 0 and nums[l] == target:
            l-=1
            
        while r< n and nums[r] == target:
            r+=1
        return [l+1,r-1]
                            
        
so = Solution()
inp = [1],1
re = so.__getattribute__(dir(so)[-1])(*inp);print re
        


class Solution(object):
    @reuse
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1:
            return 1
        r = 0
        for i in range(n):
            r += self.numTrees(i) * self.numTrees(n-1-i)
        return r
        
        
#so = Solution()
#inp = 5,
#re = so.__getattribute__(dir(so)[-1])(*inp);print re
        

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root :
            return []
        re = [[root.val]]
        def f(l):
            r = filter(None,reduce(lambda a,b:a+b, [[i.left,i.right] for i in l],[]))
            if len(r):
               re.append([i.val for i in r])
               f(r)
        f([root])
        return [sum(l)*1./len(l) for l in re]
#
#so = Solution()
#inp = TreeNode([3,9,20,null,null,15,7]),
#re = so.__getattribute__(dir(so)[-1])(*inp);print re


class Solution(object):
    def postorderTraversal(self, r, d=None):
        if d is None:
            d = []
        s = []
        while r or len(s):
            if r:
                while r.right:
                    s += [(r,1)]
                    r = r.right
#                r = s[-1]
                s += [(r,0)]
                r = r.left
            else:
                r,b = s.pop()
                d.append(r.val)
                if not s:
                    break
                if s[-1][1]:
                    r = s[-1][0].left
                    s[-1] = (s[-1][0], 0)
                else:
                    r = None
        return d 
    
#so = Solution()
#inp = TreeNode([1,2,2,3,3,null,null,4,4]),
#re = so.__getattribute__(dir(so)[-1])(*inp);print re
class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        maxx = [root.val]
        def f(r,code=''):
            if not r:
                return 0
            l,rr = f(r.left,code+'0'),f(r.right,code+'1')
            v = max(l,rr) + r.val
            v2 = l+rr+r.val
            if v2 > maxx[0]:
                maxx[0] = v2
            if v > maxx[0]:
                maxx[0] = v
            return v
        f(root, '')
        return maxx[0]
        
#        
#so = Solution()
#inp = TreeNode([1,2,3]),
#re = so.__getattribute__(dir(so)[-1])(*inp);print re
        
        
class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        l = range(1,n+1)
        def ff(li,i):
            if not len(li):
                return [None]
            lefts = f(li[:i])
            rights = f(li[i+1:])
            ts = []
            for l in lefts:
                for r in rights:
                    t = TreeNode(li[i])
                    t.left = l
                    t.right = r
                    ts.append(t)
            return ts
            
        def f(l):
            n = len(l)
            if not n:
                return[ None]
            ts = []
            for i in range(n):
                t = ff(l,i)
                ts.extend(t)
            return ts
                
        ts = f(l)
        return ts
#so = Solution()
#inp = 3,
#re = so.__getattribute__(dir(so)[-1])(*inp);print re

class Solution(object):
    def inorderTraversal(self, r, d=None):
        if d is None:
            d = []
        if not r :
            return d
        self.inorderTraversal(r.left,d)
        d.append(r.val)
        self.inorderTraversal(r.right,d)
        return d 
    def inorderTraversal(self, r, d=None):
        if d is None:
            d = []
        s = []
        while r or len(s):
            if r:
                while r.left:
                    s += [r]
                    r = r.left
            else:
                r = s.pop()
            d.append(r.val)
            r = r.right
        return d 
        
#        
#so = Solution()
#inp = TreeNode([1,2,2,3,3,null,null,4,4]),
#re = so.__getattribute__(dir(so)[-1])(*inp);print re
#        

class Solution(object):
    @reuse((3,4))
    def maxDepth(self, root, d=0,code=''):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return d
        return max(self.maxDepth(root.left, d+1,code+'0'),self.maxDepth(root.right, d+1,code+'1'),)
    
    def isBalanced(self, r,code=''):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if r is None:
            return True
        dl = self.maxDepth(r.left, 0,code+'0')
        dr = self.maxDepth(r.right, 0,code+'1')
        if abs(dl-dr) > 1:
            return False
        return True and self.isBalanced(r.left,code=code+'0') and self.isBalanced(r.right,code=code+'1')
        
#so = Solution()
#inp = TreeNode([1,2,2,3,3,null,null,4,4]),
#re = so.isBalanced(*inp);print re

class Solution(object):
    def maxDepth(self, root, d=0):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return d
        return max(self.maxDepth(root.left, d+1),self.maxDepth(root.right, d+1),)
        
#so = Solution()
#inp = TreeNode([1,2,2,3,4,4,3]),
#re = so.maxDepth(*inp);print re
class Solution(object):
    def isSymmetric(self, l,r='None'):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if r == 'None' :
            if not l:return True
            l,r = l.left, l.right
        if l is None or r is None :
            return l == r
        if l.val != r.val:
            return False
        return self.isSymmetric(l.left, r.right) and self.isSymmetric(l.right ,r.left) 

##
#so = Solution()
#inp = TreeNode([1,2,2,3,4,4,3]),
#re = so.isSymmetric(*inp);print re


class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        
        n = len(matrix)
        to = lambda v:v-n/2.+.5
        fr = lambda v:int(v+n/2.-.5)
        self.s = s = set()
        for i in range(n):
            for j in range(n):
                x,y = to(j),to(i)
                print (x,y),s
                if (x,y) in s:
                    continue
                s.add((-y,x))
#                p()
                matrix[fr(y)][fr(x)], matrix[fr(x)][fr(-y)] = matrix[fr(x)][fr(-y)],matrix[fr(y)][fr(x)]
#                break
#            break
        
#so = Solution()
#inp = [
#  [1,2,3],
#  [4,5,6],
#  [7,8,9]
#],
#re = so.rotate(*inp);print inp
        
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        def pair(l,r):
            paired = False
            while l >= 0 and r < n and s[l]=='(' and s[r]==')':
                tag[l] = tag[r] = True
                paired = True
                l-=1
                r+=1
            return paired
        n = len(s)
        tag = [False]* (n+1)
        paired = False
        for i in range(n-1):
            l, r = i, i+1
            paired |= pair(l, r)
        if not paired:
            return 0
        def getEdge():
            ps = []
            l = -1
            for i in range(n):
                if tag[i+1] is True and tag[i] is False:
                    l = i
                if  tag[i] is True and tag[i+1] is False:
#                    if l:
                    ps.append((l,i+1))
            return ps
        while paired:
            ps = getEdge()
            paired = False
            for p in ps:
                paired |= pair(*p)
#        yl.p()
        return max([r-l-1 for l,r in ps])
                    
                    
#so = Solution()
#inp = ")()())",
#re = so.longestValidParentheses(*inp);print re
        
class Solution(object):
#    @logf
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """        
        neg = (dividend>0) ^ (divisor>0)
        dividend, divisor = abs(dividend), abs(divisor)
        a,b = dividend, divisor
        if b>a:
            return 0
        d = 1
        while b < a:
            d = d << 1
            b = b << 1
        if b == a:
            r = d
        else:
            r = self.divide(a-(b>>1), divisor)+(d>>1)
        if neg:
            r = -r
        return max([min([r,2147483647]),-2147483648])
#so = Solution()
#inp = -9, 3
#re = so.divide(*inp);print re

class Solution(object):
    def reverseKGroup(self, head, k):
        last = root = ListNode(0)
        root.next = head
        while 1:
            l = []
            cur = last.next
            for i in range(k):
                if not cur:
                    break
                l.append(cur)
                cur = cur.next
            if len(l) == k:
                l[0].next = l[k-1].next
                for i in range(1, k)[::-1]:
                    l[i].next = l[i-1]
                last.next = l[-1]
                last = l[0]
            else:
                last.next = l[0] if len(l) else None
                break
        return root.next
#        
#so = Solution()
#inp = ListNode([1,2,3,4,5]), 3
#re = so.reverseKGroup(*inp);print re

class Solution(object):
    def swapPairs(self, head):
        cur = head
        l = root = ListNode(0)
        while cur and cur.next:
            l.next = cur.next
            cur.next.next, cur.next = cur, cur.next.next
            l = cur
            cur = cur.next
        if cur:
            l.next = cur
        return root.next
#so = Solution()
#inp = ListNode([1,]),
#print so.swapPairs(*inp)

class Solution(object):
    def removeNthFromEnd(self, head, n):
        cur = head
        l = []
        while cur:
            l.append(cur)
            cur = cur.next
        if len(l) == 1:
            return 
        if len(l) == n:
            return l[1]
        l[-n-1].next = None if n == 1 else l[-n+1]
        return head
#so = Solution()
#inp = ListNode([1,2]), 1
#print so.removeNthFromEnd(*inp)
def depos(n):
    if n == 0:
        return []
    l = []
    for i in range(1, n+1):
        for x in depos(n-i):
            l.append([i] + x)
    l.append([n])
    return l
def tree1(n):
    l = depos(n)
    for i in range(len(l)):
        if l[i] != 1 and isinstance(l[i],int):
            l[i] =tree1(l[i]-1)
    return l
#tree - tree1(4)
#tree- depos(4)  
class Solution(object):
    def generateParenthesis(self, deep):
        l = []
        def f(notin, inl=0, inr=0, s=''):
            if notin :
                f(notin-1, inl+1, inr, s+'(')
            if inl > inr :
                f(notin, inl, inr+1, s+')')
            if not notin and inl == inr:
                l.append(s)
        f(deep,)
        return l
#so = Solution()
#inp = 3
#tree- so.generateParenthesis(inp)
class Solution(object):
    def letterCombinations(self, digits):
        d = {'3': 'def', '2': 'abc', '5': 'jkl', '4': 'ghi', '7': 'pqrs', '6': 'mno', '9': 'wxyz', '8': 'tuv'}
        l = []
        def f(di,s=''):
            if not di:
                l.append(s)
                return 
            for i in d[di[0]]:
                f(di[1:],s+i)
        f(digits)
        return l

#
#so = Solution()
#inp = '23'
#print so.letterCombinations(inp)
#reuse = lambda x:x
class Solution(object):
    def maxArea(self, height):
        maxx = 0
        n = len(height)
        for i in range(n-1):
            for j in range(i+1,n):
                l,r = height[i],height[j]
                now = min([l,r]) * (j-i)
                maxx = max(now, maxx)
        return maxx
    def maxArea(self, height):
        maxx = 0
        n = len(height)
        l,ll,r,rr = 0,0,n-1,n-1
        while r-l>0:
            lv, rv = height[l], height[r]
            maxx = max([maxx,min([lv,rv])*(r-l)])
            if lv > rv:
                r -= 1
            else:
                l += 1
#            print l,r,lv,rv,[maxx,min([lv,rv])*(r-l)]
        return maxx
#    
#so = Solution()
#inp = [1,30,33,1]
#print so.maxArea(inp)
        
        
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = last = ListNode(0)
        while 1:
            if l1 is None:
                last.next = l2
                break
            if l2 is None:
                last.next = l1
                break
            if l1.val > l2.val:
                l1,l2 = l2,l1
            last.next = l1
            l1,last = l1.next, l1
        return head.next
    
    def insertionSortList(self, head):
        '''归并排序（MERGE-SORT） O(n log(n))'''
        q = []
        while head:
            q.append(head)
            head.next, head= None, head.next
        while len(q)>1:
            a,b = q.pop(),q.pop()
            r = self.mergeTwoLists(a, b)
            q = [r] + q
        return q and q[0]
    def insertionSortListOld(self, head):
        h = ListNode(0)
        while head:
            l = h
            next = head.next
            while l.next and head.val > l.next.val:
                l = l.next
            head.next, l.next =  l.next, head    
            head = next
        return h.next
#ns = ListNode([2,9,6,3,1,8])

#so = Solution()
#print so.insertionSortList(ns)

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = last = BiList(0)
        while 1:
            if l1 is None:
                last.n = l2
                break
            if l2 is None:
                last.n = l1
                break
            if l1.v.val > l2.v.val:
                l1,l2 = l2,l1
            last.n = l1
            l1,last = l1.n, l1
        return head.n
    
    def insertionSortList(self, q):
        '''归并排序（MERGE-SORT） O(n log(n))'''
        while len(q)>1:
            a,b = q.pop(),q.pop()
            r = self.mergeTwoLists(a, b)
            q = [r] + q
        return q and q[0]
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        k = len(lists)
        if not k:
            return []
        head = last = ListNode(0)
        f = lambda x:x.val
        q = map(lambda x:BiList(x,f) ,filter(None, lists))
        nodes = self.nodes = BiList(head,keyFun=f)
        nodes.n = self.insertionSortList(q)
        while nodes.n:
            nodes.n.b = nodes
            nodes = nodes.n
        nodes = nodes.head
#        return 
        while nodes.n:
            last.next = nodes.n.v
            nodes.delt(nodes.n)
            if last.next.next:
                nodes.insert(last.next.next)
            last = last.next
        return head.next

#ns = map(ListNode,[[0,3],[1,2,4],[9]])
#ns = map(ListNode,
#ns = ns[:10]
#so = Solution()
#crun('so.mergeKLists(ns)')
#heatMap('so.mergeKLists(ns)')
#res= so.mergeKLists(ns)
#print res        
#a = BiList(r)
#a.insert(4)
#a.insert(6)
        
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = last = ListNode(0)
        while 1:
            if l1 is None:
                last.next = l2
                break
            if l2 is None:
                last.next = l1
                break
            if l1.val > l2.val:
                l1,l2 = l2,l1
            last.next = l1
            l1,last = l1.next, l1
        return head.next



class Solution(object):
    @reuse
    def find2(self, v):
        ns = self.nums
        n = len(ns)
        b, e = 0, n
        while 1:
            if b>=e:
                return None
            m = (e+b)//2
            vv = ns[m]
            if v == vv:
                while v == ns[m-1]:
                    m -= 1
                return m
            elif v >vv:
                b,e = m+1,e
            else:
                b,e = b,m
            
    def add(self, a,b,c):
        t = tuple(sorted([a,b,c]))
        self.s.add(t)
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.s = set()
        nums.sort()
        n = len(nums)
        if not n :
            return []
        ns = self.nums = nums
        pos = 0
        while pos<n and ns[pos] < 0 :
            pos += 1
        if pos+2< n  and ns[pos]==ns[pos+1]==ns[pos+2]==0:
            self.add(pos,pos+1,pos+2)
        for i in range(pos):
            if i>0 and ns[i-1]==ns[i]:
                continue
            for j in range(pos,n):
                if j>0 and ns[j-1]==ns[j]:
                    continue
                v = -(ns[i]+ns[j])
                ind = self.find2(v)
                if ind is None:
                    pass
                elif ind == i:
                    if i+1 <n and ns[i+1]==ns[i]:
                        self.add(i,i+1,j)
                elif ind == j: 
                    if j+1 <n and ns[j+1]==ns[j]:
                        self.add(i,j,j+1)
                else:
                    self.add(i,ind,j)
#        p()
        return [[ns[i] for i in t] for t in self.s]
                    
                    
#so = Solution()
#l = [-1, 0, 1, 2, -1, -4]
#l = range(-1000,1000)
#l = [-1, 0, 1,0]
#a= so.threeSum( l)
#print a

class Solution(object):
    def f(l,r):
        l
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        l = height
        n = len(l)
        for i in range(n):
            for j in range(i,n):
                pass
                    
#so = Solution()
#print so.maxArea([1,1,1,1])

class Solution(object):
    def push(self,b,e):
        if e - b < self.bed[2]:
            return 
        self.bed = (b,e,e-b)
        
    def f(self, s, b=0, e=-1):
        """
        :type s: str
        :rtype: str
        """
        if e == -1:
            self.bed = (0,1,1)
            e = len(s)
            self.d = {}
        d = self.d
        if (b,e) in d:
            return d[(b,e)]
        if e-b <= 1:
            return 1
        if e == s:
            return 1
            
        if s[b] == s[e-1]:
            inn = self.f(s, b+1,e-1)
            if inn :
                self.push(b,e)
                d[(b,e)] = True
                return True
        [self.f(s, b+1,e),self.f(s,b,e-1) ]
        d[(b,e)] = False
        return False
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        self.f(s,)
        b,e,d = self.bed
        return s[b:e]

class Solution(object):
    def longestPalindrome(self, s):
        ss = s[0]
        maxx = 1
        n = len(s)
        for i in range(1,n):
#            if i == s[i-1]:
#            l,r,d = i-1,i,0
            for l,d in [(i-1,0), (i-2,1)]:
                r = i
#            if i == s[i-2]
#                l,r,d = i-2,i,1
                while  l>=0 and r<n and s[l] == s[r]:
                    d += 2
                    l -= 1
                    r += 1
                if d > maxx:
                    maxx = d
                    ss = s[l+1:r]
        return ss
            
                
            
            
#so = Solution()
#        
#print so.longestPalindrome("bb")
        
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        sn = ''
        lenn = len(s)
        n = numRows
        gap = max([1,2*n-2])
        l = [0] 
        for i in range(n-2):
            l += [[i+1,gap-1-i]]
        if n > 1:
            l += [n-1]
        for i in l:
            if isinstance(i, int):
                ind = i
                while ind < lenn:
                    sn += (s[ind])
                    ind += gap
            else :
                ind1,ind2 = i
                while 1:
                    b1,b2 =  ind1 < lenn, ind2 < lenn
                    if b1:
                        sn += (s[ind1])
                        ind1 += gap
                    if b2:
                        sn += (s[ind2])
                        ind2 += gap
                    if not(b1 or b2):
                        break
        return sn
#so = Solution()
#print so.convert('A',1)
#print so.convert('ABCD',3)


class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        import re
        p = re.compile('^[+-]?[0-9]+')
        l = p.findall(str.strip())
        if not l:
            return 0
        return min([max([int(l[0]), -2147483648]),2147483647])
        
        if '.' in str:
            return 0
        try :
            intt = int(str)
        except Exception:
            return 0
        
        return min([max([intt, -2147483648]),2147483647])















