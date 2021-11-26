# -*- coding: utf-8 -*-
import sys
from opencc import OpenCC
import io

def chs_to_cht(src, dst):
    cc = OpenCC('s2twp')
    with io.open(src) as f:
        input_str = f.read()
        output_str = cc.convert(input_str)
        with io.open(dst, 'w', encoding='UTF-8') as f:
            f.write(output_str)

if __name__ == '__main__':
    chs_to_cht(sys.argv[1], sys.argv[2])