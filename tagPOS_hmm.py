#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import csv, sys, re 

dataPath = 'WSJ_POS_CORPUS_FOR_STUDENTS/'
threshold = 1
suflen = 2
lam = 0.3

class POStagger_HMM(object):
    def __init__(self):
        self.Pemit = {}     # { Pos : { word : Pemit }}
        self.Words = {}     # { word : { Pos : count} ) }
        self.PosSize = 0 
        # self.PosCount = {}
        # self.unkonwnCount = 0   
        self.suffix = {}
        self.openClass = set([':','CD','FW','IN','JJ','JJR','JJS','NN',\
                 'NNP','NNPS','NNS','RB','RBR','RBS','UH',\
                 'VB','VBD','VBG','VBN','VBP','VBZ','SYM'])
        self.morphCatNum = 16  # normal; Cap; multiCap; 
                            # hyphen; hyphen+Cap; digit; number
        # self.tag  [Pos]
        # self.label { Pos : enum }
        # self.unknown  [ P(unknown|Pos) ]  # hapax legomena model
    
    def morphCat(self,word):
        if re.match('\A[a-zA-Z]+\Z',word):
        # pure words
            if re.match('\A[a-z]+\Z',word):
            # all lower case word
                return 0
            elif re.match('\A[A-Z][a-z]*\Z',word):
            # Cap word
                return 1
            elif re.match('\A[A-Z]+\Z',word):
                return 2
            else:
                return 3
        elif '-' in word:
        # hyphen 
            if re.match('\A[A-Z][^-]*-[A-Z].*\Z',word):
            # Cap-Cap
                return 4
            elif re.match('\A\d\d?\d?(,?\d\d\d)*(.\d*)?-.*\Z',word):
            # number-seq 
                return 5
            elif re.match('\A\D+-\d+\Z',word):
            # seq-number
                return 6
            elif re.match('\A[a-z]+-[A-Z].*\Z',word):
                return 7
            elif re.match('\A[A-Z]+-[a-z].*\Z',word):
                return 15
            elif word.count('-') > 1:
                return 8
            else:
                return  9
        elif re.search('\d',word):
        # digits 
            if re.match('\A[+-]?\d\d?\d?(,?\d\d\d)*(.\d*)?\Z',word):
            # number
                return 10
            else:
                return 11
        elif '\/' in word: 
            return 12
        elif '.' in word:
            return 13
        else:
            return 14

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
            exit()
        # read the file once and ...
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
            # word POS
            if word not in self.Words:
                self.Words[word] = { Pos : 1 }
            elif Pos not in self.Words[word]:
                self.Words[word][Pos] = 1
            else:
                self.Words[word][Pos] += 1
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
        ftrain.close()

        self.PosSize = len(self.Pemit)
        self.label = {Pos:enum for enum, Pos in enumerate(self.Pemit)}
        tmp = [(self.label[Pos],Pos) for Pos in self.label]
        tmp.sort()
        self.tag = [t[1] for t in tmp]
        self.tag.append('START')
        self.label.update({'END':self.PosSize,'START':self.PosSize})
        
        # suffix model
        for word in self.Words:
            numsuf = suflen if len(word)>=suflen else len(word)
            sufl = [word[-i:] for i in range (1,numsuf+1)] 
            for suf in sufl:
                if suf not in self.suffix:
                    self.suffix[suf] = np.zeros(self.PosSize+1)
                for Pos in self.Words[word].keys():
                    self.suffix[suf][self.label[Pos]] += 1 #self.Words[word][Pos]
        for suf in self.suffix.keys():
            total = np.sum(self.suffix[suf])
            if total >= 5:
                self.suffix[suf] += 1
                self.suffix[suf] *= 1./np.sum(self.suffix[suf])
                #self.suffix[suf] = np.log2(self.suffix[suf])
            else:
                self.suffix.pop(suf)
        # morphological model
        self.morph = np.zeros([self.morphCatNum,self.PosSize+1])
        for word in self.Words:
            cat = self.morphCat(word)
            if cat == 9:
                print word 
            for Pos in self.Words[word].keys():
                if cat > 1:
                    self.morph[cat,self.label[Pos]] += 1
                else:
                    if self.Words[word][Pos] <= threshold:
                        self.morph[cat,self.label[Pos]] += 1
        print np.sum(self.morph,axis=1)
        for vec in self.morph:
            vec = vec+1
            vec /= np.sum(vec)
            #vec = np.log2(vec)

        self.unknown = np.zeros(self.PosSize+1)
        # transition probabilities
        self.TransMat = np.zeros([self.PosSize+1,self.PosSize+1])
        for PosPre in Ptrans:
            i = self.label[PosPre]
            for Pos in Ptrans[PosPre]:
                self.TransMat[i,self.label[Pos]] = Ptrans[PosPre][Pos] 
        for vec in self.TransMat:
            vec += 1    # smooth
            vec /= np.sum(vec)
            #vec = np.log2(vec)
        self.TransMat = self.TransMat.T     # For cache friendliness
        # emission probabilities
        for Pos in self.Pemit:
            vec = np.array(self.Pemit[Pos].values())
            total = np.sum(vec)
            if Pos in self.openClass:
                self.unknown[self.label[Pos]] = np.sum(vec <= threshold)*1./total
                # self.PosCount[Pos] = total
            for word in self.Pemit[Pos]:
                self.Pemit[Pos][word] *= 1./total
                #self.Pemit[Pos][word] = np.log2(self.Pemit[Pos][word])
        self.unknown /= np.sum(self.unknown)
        #self.unknown = np.log2(self.unknown)


    def getPosTransEmit(self,word):
        ret = []
        if word in self.Words:
            for Pos in self.Words[word].keys():
                ret.append((Pos,self.Pemit[Pos][word]))
        else:
            cat = self.morphCat(word)
            flag = 0
            numsuf = suflen if len(word)>=suflen else len(word)
            sufl = [word[-i:] for i in range (numsuf,0,-1)] 
            for suf in sufl:
                if suf in self.suffix:
                    ret = [(Pos,emit) for (Pos,emit) in zip(self.tag,self.unknown*self.suffix[suf]*self.morph[cat])]
                    flag = 1
                    break
            if flag == 0:
                ret = [(Pos,emit) for (Pos,emit) in zip(self.tag,self.unknown*self.morph[cat])]
            
            if cat < 4:
                if word.lower() in self.Words: 
                    total = sum(self.Words[word.lower()].values())
                    for Pos in self.Words[word.lower()]:
                        ret[self.label[Pos]] = (Pos,ret[self.label[Pos]][1] + self.Words[word.lower()][Pos]*1./total)
            
        return ret 

    def tagSentence(self,snt):
        T = len(snt)
        Vtb = np.zeros([T+2,self.PosSize+1])
        Trace = np.ones([T+2,self.PosSize+1])*-1
        Vtb[0,self.label['START']] = 1
        # Vtb = np.log2(Vtb)
        ret = []
        for i in range(1,T+1):
            # Vtb[i,self.label[Pos]] = np.max( Vtb[i] * trans * emit )
            word = snt[i-1]
            PosSet = self.getPosTransEmit(word)
            # print PosSet
            for Pos,emit in PosSet:
                trans = self.TransMat[self.label[Pos]]
                tmp = Vtb[i-1] * trans * emit 
                # tmp = Vtb[i-1] + trans + emit
                Vtb[i,self.label[Pos]] = np.max(tmp) *10
                Trace[i,self.label[Pos]] = np.argmax(tmp)

        i = T+1
        trans = self.TransMat[self.label['END']]
        tmp = Vtb[i-1] * trans
        # tmp = Vtb[i-1] + trans
        Vtb[i,self.label[Pos]] = np.max(tmp)
        Trace[i,self.label[Pos]] = np.argmax(tmp)
        lastPos = np.argmax(tmp)
        for i in range (T,0,-1):
            # print self.tag[lastPos]
            ret.append(self.tag[lastPos])
            lastPos = int(Trace[i,lastPos])
        ret.reverse()
        return ret     

    def tagFile(self,path,fout):
        fin = open(path,'r')
        word = fin.readline()
        snt = []
        while word != '':
            word = word.strip('\n')
            if word != '':
                snt.append(word)
            else:
                map(fout.write,["%s\t%s\n"%(x,y) for (x,y) in zip(snt,self.tagSentence(snt))])
                snt = []
                fout.write("\n")
            word = fin.readline()
        fin.close()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        filePath = 'WSJ_02-21.pos'
    else:
        filePath = sys.argv[1]
    path = dataPath + filePath
    tagger = POStagger_HMM()
    tagger.train(path)
    # print tagger.label
    if len(sys.argv) < 3:
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
    else: 
        fout = open(sys.argv[2]+".pos","w")
        tagger.tagFile(dataPath+sys.argv[2]+".words",fout)
        fout.close()
    # print tagger.label
    # print tagger.unkonwnCount
    # print tagger.unknown
    # print filter(lambda (x,y):y!=0,zip(tagger.tag,tagger.unknown))
