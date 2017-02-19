#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import csv, sys

dataPath = 'WSJ_POS_CORPUS_FOR_STUDENTS/'

class POStagger_HMM(object):
    def __init__(self):
        self.Pemit = {} # {Pos:{word:count}}
        self.Ptrans = {} # {Pos:{Pos:count}}

    def readFile(self,path):
        PosPre = 'START'
        for line in csv.reader(file(path),delimiter='\t'):
            if len(line) == 0:

                continue
            word,Pos = line[0],line[1]
            if Pos not in self.Pemit:
                self.Pemit[Pos] = {}
            if word not in self.Pemit[Pos]:
                self.Pemit[Pos][word] = 0
            self.Pemit[Pos][word] += 1



if __name__ == '__main__':
    if len(sys.argv) == 1:
        filePath = 'WSJ_02-21.pos'
    else:
        filePath = sys.argv[1]
    path = dataPath + filePath
    tagger = POStagger_HMM()
    tagger.readFile(path)
        