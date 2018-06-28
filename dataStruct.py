#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:01:22 2018

@author: dl
"""

from yllab import *
import types
null = None
def logf(f):
    def ff(*l, **kv):
        r = f(*l, **kv)
        tree([l,r])
        return r
    return ff

def reuse(s=(0,None)):
    d = {}
    f = [s]
    def innerF(*l, **kv):
        
        k = l[s]
        if k not in d:
            d[k] = f[0](*l, **kv)
        return d[k]
    def decorate(ff):
        f[0] = ff
        return innerF
    if isinstance(s, types.FunctionType):
        s = slice(0,None)
        return innerF
    s = slice(*s)
    return decorate
def listToBatch(listt, batch):
    '''
    将一段序列按照每batch个元素组成一组
    >>> listToBatch(range(8),3)
    [(0, 1, 2), (3, 4, 5), (6, 7)]
    '''
    n = len(listt)
    left = n % batch
    if left:
        ind  = n - left
        listt, tail = listt[:ind], tuple(listt[ind:])
    ziped = zip(*[iter(listt)]*batch)
    if left:
        ziped.append(tail)
    return ziped


def ttod(n,f=None,d='t'):
    if f is None:
        f = {}
    if n is None:
        return 
    k = d,n.val
    f[k]={}
    ttod(n.left,f[k],'l')
    ttod(n.right,f[k],'r')
    return f
def ttol(r):
    if r:
        return [r.val, ttol(r.left), ttol(r.right)]
class TreeNode(dicto):
    def __init__(self, x):
         if isinstance(x,list):
             TreeNode.init(self, x)
             return 
         self.val = x
         self.left = None
         self.right = None
    def init(self, l):
        b = 1
        self.__init__(l[0])
        level = [self]
        while len(level) and b < len(l):
#            print [i.val for i in level]
            r = level.pop()
            if l[b]:
                r.left = TreeNode(l[b])
                level = [r.left] + level
            if l[b+1]:
                r.right = TreeNode(l[b+1])
                level = [r.right] + level
            b += 2
#    def __str__(self):
#        tree-self
#        return ''
#    __repr = __str__
    
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x, next=None):
         if isinstance(x, list):
             while len(x)>1:
                 next = ListNode(x.pop(),next)
             x = x.pop()
         self.val = x
         self.next = next
    def __str__(self,):
        s = ''
        while self:
            s+= '%s->'%((self.val),)
            self = self.next
        return (s)
    __repr__ = __str__

from collections import namedtuple

class BiList():
    def __init__(self, v=None, keyFun=None, n=None, b=None):
        self.v = v
        self.keyFun = keyFun or (lambda x:x)
        self.n = n
        self.b = b
    def insert(self, v, keyFun=None):
        if keyFun is None:
            keyFun = self.keyFun
        last = self
        while last.n is not None and keyFun(v)>keyFun(last.n.v):
            last = last.n
        new = BiList(v, keyFun, last.n, last)
        if last.n is not None:
            last.n.b = new
        last.n = new
    def delt(self, node):
        if node.b:
            node.b.n = node.n
        if node.n:
            node.n.b = node.b
    @property
    def head(self):
        while self.b:
            self = self.b
        return self
    def __str__(self,):
        s = ''
#        self = self.head
        while self:
            s+= '%s<->'%(self.keyFun(self.v),)
            self = self.n
        return (s)
    __repr__ = __str__

