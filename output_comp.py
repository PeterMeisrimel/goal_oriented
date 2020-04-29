#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:17:17 2018

@author: Peter Mesirimel, Lund University
"""

from __future__ import division

file = 'output.txt'
file_ref = 'output_ref.txt'

ref, out = [], []


with open(file_ref, 'r') as myfile:
    for line in myfile:
        ref.append(line)

with open(file, 'r') as myfile:
    for line in myfile:
        out.append(line)
        
for i, line in enumerate(out):
    if line.find('second') != -1:
        continue
    if(line != ref[i]):
        print("Failed,\n {}\n {} {}".format(i+1, line, ref[i]))
    
print('DONE')