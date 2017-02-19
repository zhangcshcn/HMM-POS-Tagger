#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import csv, sys

dataPath = 'WSJ_POS_CORPUS_FOR_STUDENTS/'

class POStagger_HMM(object):
    def __init__(self):
        self.Pemit = {}     # { Pos : { word : count }}
        # Ptrans = {}    # { Pos : { Pos : count }}
        self.Words = {}     # { word : set( Pos ) }
        self.PosSize = 0 

    # sentence model: 
    #   START => { 45 POSes } => END
    # START has a transtion probability. 
    # Words are case sensitive. 
    def train(self,path):
        PosPre = 'START'
        Ptrans= { PosPre:{} }
        self.path = path
        try:
            ftrain = open(path,'r')
        except:
            print 'Error opening the file... Please try again... '
        # read the file once and 
        for line in csv.reader(ftrain,delimiter='\t'):
            if len(line) == 0:
                Pos = 'END'
                if PosPre not in Ptrans:
                    Ptrans[PosPre] = {Pos:1}
                elif Pos not in Ptrans[PosPre]:
                    Ptrans[PosPre][Pos] = 1
                else:
                    Ptrans[PosPre][Pos] += 1
                PosPre = 'START'
                continue            
            word,Pos = line[0],line[1]

            if word not in self.Words:
                self.Words[word] = set(Pos)
            else:
                self.Words[word] |= set(Pos)
            # emission count
            if Pos not in self.Pemit:
                self.Pemit[Pos] = {word:1}
            elif word not in self.Pemit[Pos]:
                self.Pemit[Pos][word] = 1
            else:
                self.Pemit[Pos][word] += 1

            # transition count
            if PosPre not in Ptrans:
                Ptrans[PosPre] = {Pos:1}
            elif Pos not in Ptrans[PosPre]:
                Ptrans[PosPre][Pos] = 1
            else:
                Ptrans[PosPre][Pos] += 1
            PosPre = Pos

        self.PosSize = len(self.Pemit)
        self.label = {Pos:enum for enum, Pos in enumerate(self.Pemit)}
        self.label.update({'END':self.PosSize,'START':self.PosSize})
        # transition probabilities
        self.TransMat = np.zeros([self.PosSize+1,self.PosSize+1])
        for PosPre in Ptrans:
            i = self.label[PosPre]
            for Pos in Ptrans[PosPre]:
                self.TransMat[i,self.label[Pos]] = Ptrans[PosPre][Pos] 
        for vec in self.TransMat:
            vec /= np.sum(vec)
        # emission probabilities
        for Pos in self.Pemit:
            total = sum(self.Pemit[Pos].values())
            for word in self.Pemit[Pos]:
                self.Pemit[Pos][word] *= 1./total


    def tagSentence(self,path):
        pass
        


if __name__ == '__main__':
    if len(sys.argv) == 1:
        filePath = 'WSJ_02-21.pos'
    else:
        filePath = sys.argv[1]
    path = dataPath + filePath
    tagger = POStagger_HMM()
    tagger.train(path)

    #sentece = input('Enter a sentence.')
    '''
    for Pos in tagger.Ptrans:
        print sum(tagger.Ptrans[Pos].values())        

    for Pos in tagger.Pemit:
        print sum(tagger.Pemit[Pos].values())
    '''