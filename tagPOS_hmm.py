#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import csv, sys

dataPath = 'WSJ_POS_CORPUS_FOR_STUDENTS/'

class POStagger_HMM(object):
    def __init__(self):
        self.Pemit = {}     # { Pos : { word : Pemit }}
        self.Words = {}     # { word : set( Pos ) }
        self.PosSize = 0 

    # sentence model: 
    #   START => { 45 POSes } => END
    # START has a transtion probability. 
    # Words are case sensitive. 
    def train(self,path):
        PosPre = 'START'
        Ptrans= { PosPre:{} }   # { PosPre: { Pos : count }}
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
                self.Words[word] = set([Pos])
            else:
                self.Words[word] |= set([Pos])
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
        tmp = [(self.label[Pos],Pos) for Pos in self.label]
        tmp.sort()
        self.tag = [t[1] for t in tmp]
        self.tag.append('START')
        self.label.update({'END':self.PosSize,'START':self.PosSize})
        # transition probabilities
        self.TransMat = np.zeros([self.PosSize+1,self.PosSize+1])
        for PosPre in Ptrans:
            i = self.label[PosPre]
            for Pos in Ptrans[PosPre]:
                self.TransMat[i,self.label[Pos]] = Ptrans[PosPre][Pos] 
        for vec in self.TransMat:
            vec /= np.sum(vec)
        self.TransMat = self.TransMat.T     # For cache friendliness
        # emission probabilities
        for Pos in self.Pemit:
            total = sum(self.Pemit[Pos].values())
            for word in self.Pemit[Pos]:
                self.Pemit[Pos][word] *= 1./total
        fp = open("Words","w")
        '''
        for word in self.Words:
            fp.write("%s"%word)
            map(fp.write,["\t%s"%(pos) for pos in self.Words[word]])
            fp.write("\n")
        fp.close()
        '''

    def getPos(self,word):
        if word in self.Words:
            return self.Words[word]
        else:
            print word, "unknown"

    def tagSentence(self,snt):
        T = len(snt)
        Vtb = np.zeros([T+2,self.PosSize+1])
        Trace = np.zeros([T+2,self.PosSize+1])
        Vtb[0,self.label['START']] = 1
        ret = []
        for i in range(1,T+1):
            # Vtb[i,self.label[Pos]] = np.max( Vtb[i] * trans * emit )
            word = snt.pop(0)
            PosSet = self.getPos(word)
            for Pos in PosSet:
                emit = self.Pemit[Pos][word]
                trans = self.TransMat[self.label[Pos]]
                tmp = Vtb[i-1] * trans * emit
                Vtb[i,self.label[Pos]] = np.max(tmp)
                Trace[i,self.label[Pos]] = np.argmax(tmp)
        i = T+1
        tmp = Vtb[i-1] * trans * emit
        Vtb[i,self.label[Pos]] = np.max(tmp)
        Trace[i,self.label[Pos]] = np.argmax(tmp)
        lastPos = np.argmax(tmp)
        for i in range (T,0,-1):
            ret.append(self.tag[lastPos])
            lastPos = int(Trace[i,lastPos])
        ret.reverse()
        # print Trace
        return ret      

if __name__ == '__main__':
    if len(sys.argv) == 1:
        filePath = 'WSJ_02-21.pos'
    else:
        filePath = sys.argv[1]
    path = dataPath + filePath
    tagger = POStagger_HMM()
    tagger.train(path)

    while True:
        try:
            sentence = input('Enter a sentence: ')
            #sentence = "I don't know what he is talking about !"
            snt = sentence.split(' ')
            try:
                tag = tagger.tagSentence(snt)
                print zip(sentence.split(' '),tag)
            except:
                pass
        except:
            break
    '''
    for Pos in tagger.Ptrans:
        print sum(tagger.Ptrans[Pos].values())        

    for Pos in tagger.Pemit:
        print sum(tagger.Pemit[Pos].values())
    '''