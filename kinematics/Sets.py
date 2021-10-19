from sortedcontainers import SortedList, SortedSet
from random import random

class ContinuousRange():
    def __init__(self, start, end, start_include=True, end_include=True):
        correct = start < end  # sort the range if given out of order
        if correct:
            self.a = start
            self.b = end if correct else start
            self.a_incl = start_include
            self.b_incl = end_include
        else:
            self.a = end
            self.b = start
            self.a_incl = end_include
            self.b_incl = start_include

    @property
    def middle(self):
        return (self.a + self.b)/2

    def sample(self):
        return self.a + (self.b-self.a) * random()


    def __repr__(self):
        return '{}{}, {}{}'.format('[' if self.a_incl else '(', self.a, self.b, ']' if self.b_incl else ')')

    def __str__(self):
        return '{}{}, {}{}'.format('[' if self.a_incl else '(', self.a, self.b, ']' if self.b_incl else ')')

    def __lt__(self, other):
        lower = False
        if self.a == other.a:
            lower = self.b < other.b
        else:
            lower = self.a < other.a
        return lower

class ContinuousSet():
    def __init__(self, start=None, end=None, start_include=True, end_include=True):
        self.c_ranges = SortedList()  # set of ContinuousRanges

        if not(start==None or end==None):
            assert start!=end, 'Continuous set cannot have only one value'
            self.add_c_range(start, end, start_include=True, end_include=True)

    def assert_range(self):
        assert len(self.c_ranges) == 1, 'Make sure that only one range interval is present in the set!'

    @property
    def a(self):
        self.assert_range()
        return self.c_ranges[0].a

    @property
    def b(self):
        self.assert_range()
        return self.c_ranges[0].b

    @property
    def a_incl(self):
        self.assert_range()
        return self.c_ranges[0].a_incl

    @property
    def b_incl(self):
        self.assert_range()
        return self.c_ranges[0].b_incl

    @property
    def middle(self):
        return (self.a + self.b)/2

    @property
    def empty(self):
        return len(self.c_ranges)==0

    def sample(self):
        self.assert_range()
        return self.c_ranges[0].sample()

    @property
    def size(self):
        s = 0.0
        for cr in self:
            s += cr.b - cr.a
        return s

    def add_c_range(self, start, end, start_include=True, end_include=True):
        self.c_ranges = self._add(ContinuousRange(start, end, start_include, end_include))

    def sub_c_range(self, start, end, start_include=True, end_include=True):
        self.c_ranges = self._sub(ContinuousRange(start, end, start_include, end_include))

    def _add(self, c_range):
        c_ranges_new = self.c_ranges.copy()
        startt = c_range
        endd = c_range
        # Find the start and end overlaping intervals
        to_remove = set()
        for cr in self:
            if c_range.a < cr.a and c_range.b > cr.b:
                to_remove.add(cr)
            if cr.a <= c_range.a <= cr.b:
                startt = cr
                to_remove.add(cr)
            if cr.a <= c_range.b <= cr.b:
                endd = cr
                to_remove.add(cr)
                break

        # Remove continuous ranges in between
        for cr in to_remove:
            c_ranges_new.discard(cr)
        a_incl = startt.a_incl or c_range.a_incl if abs(startt.a-c_range.a)<1e-8 else startt.a_incl
        b_incl = endd.b_incl or c_range.b_incl if abs(endd.b-c_range.b)<1e-8 else endd.b_incl

        # Add the new range
        c_ranges_new.add(ContinuousRange(startt.a, endd.b, a_incl, b_incl))
        return c_ranges_new

    def __add__(self, other):
        ret_set = ContinuousSet()
        ret_set.c_ranges = self.c_ranges.copy()
        for cr in other:
            ret_set.c_ranges = ret_set._add(cr)
        return ret_set

    def __iter__(self):
        for c_r in self.c_ranges:
            yield c_r

    def _sub(self, new_range):
        c_ranges_new = SortedList()
        # Find the start and end overlaping intervals
        for cr in self:
            if new_range.a < cr.a and new_range.b > cr.b:
                # 1. new range includes range
                c_ranges_new.add(cr)

            if cr.a <= new_range.a < cr.b:
                # 2. start of new range is in range (and end of new range can be either in(2.1) or out(2.2))
                a_incl = cr.a_incl and new_range.a_incl if abs(cr.a - new_range.a) < 1e-6 else new_range.a_incl
                if new_range.b > cr.b:
                    # 2.1. only start of new range included in range
                    c_ranges_new.add(ContinuousRange(new_range.a, cr.b, a_incl, cr.b_incl))

                else:
                    # 2.2. new range included in range
                    b_incl = cr.b_incl and new_range.b_incl if abs(cr.b - new_range.b) < 1e-7 else new_range.b_incl
                    c_ranges_new.add(ContinuousRange(new_range.a, new_range.b, a_incl, b_incl))

            elif cr.a < new_range.b <= cr.b and new_range.a < cr.a:
                # 3. only end of new range included in range
                b_incl = cr.b_incl and new_range.b_incl if abs(cr.b - new_range.b) < 1e-7 else new_range.b_incl
                c_ranges_new.add(ContinuousRange(cr.a, new_range.b, new_range.a_incl, b_incl))

        return c_ranges_new

    def __sub__(self, other):
        ret_set = ContinuousSet()
        for o_r in other.c_ranges:
            ret_set += self._sub(o_r)
        return ret_set

    def __repr__(self):
        # return 'Set: ' + str([c_r for c_r in self.c_ranges])
        return '  Set: {' + ', '.join(map(str,self.c_ranges)) + '}  '

    def __contains__(self, key):
        self.assert_range()
        return self.c_ranges[0].a < key < self.c_ranges[0].b

    def inverse(self, lower_lim, higher_lim):
        ret_set = ContinuousSet()
        start_r = ContinuousRange(lower_lim-2, lower_lim, False, False)
        for cr in self.c_ranges:
            if cr.a > start_r.b:
                ret_set.c_ranges.add(ContinuousRange(start_r.b, cr.a, start_r.b_incl, cr.a_incl ))
                start_r = cr
            else:
                start_r = cr
        if start_r.b < higher_lim:
            ret_set.c_ranges.add(ContinuousRange(start_r.b, higher_lim, start_r.b_incl))

        return ret_set


##############################################################################

if __name__ == "__main__":

    # test the sets #TODO: add proper testing
    ss = ContinuousSet()

    ss += ContinuousSet(2,3,)
    ss.add_c_range(2.1,3.1)
    ss.add_c_range(4,15, False, False)
    ss.add_c_range(12,5)
    ss.add_c_range(4,5)
    ss.add_c_range(5,6)

    ss2 = ContinuousSet()

    ss2.add_c_range(2, 3.4, False)
    ss2.add_c_range(122, 13.9, False)
    ss2.add_c_range(2.1,3.5)
    ss2.add_c_range(4.2,13.8)
    ss2.add_c_range(5, 11, False, False)


    ss3 = ContinuousSet(2,1)
    ss4 = ContinuousSet(2,3)


    ss4.sub_c_range(2.1, 4.5, False)

    print('ss4', ss4)
    print('s34', ss3 - ss4, (ss3-ss4).empty)
    print('ss', ss)
    print('ss2', ss2)
    print('ss2_inverse06',ss2.inverse(0, 6))
    print('+', ss + ss2)
    print('-', ss - ss2)
    # print(ss)
    print((ss-ss2).size)

import numpy as np
print(ContinuousSet(-np.pi , -np.pi/2) + ContinuousSet(np.pi/2 , np.pi))