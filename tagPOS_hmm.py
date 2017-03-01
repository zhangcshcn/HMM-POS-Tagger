#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import csv, sys, re 

dataPath = 'WSJ_POS_CORPUS_FOR_STUDENTS/'
threshold = 1
suflen = 2

class POStagger_HMM(object):
    def __init__(self):
        self.Pemit = {}     # { Pos : { word : Pemit }}
        self.Words = {}     # { word : { Pos : count} ) }
        self.PosSize = 0 
        self.suffix = {}
        self.openClass = set([':','CD','FW','IN','JJ','JJR','JJS','NN',\
                 'NNP','NNPS','NNS','RB','RBR','RBS','UH',\
                 'VB','VBD','VBG','VBN','VBP','VBZ','SYM'])
        self.morphCatNum = 16  
        self.lam2 = 0
        self.lam3 = 0

    # Use regex to determine the morphological features of words
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

    # - Train the 1st and 2nd-Order Markov Models. 
    # - Use the Hapax Legomena with Open Class. Train models with  
    #   morphological and suffix features respectively. 
    def train(self,paths):
        Ptrans= { 'START':{} }   # { (Pos,Pos): { Pos : count }}
        self.paths = paths
        for path in paths:
            try:
                ftrain = open(dataPath+path,'r')
            except:
                print 'Error opening the file... Please try again... '
                exit()
            # read the file once and ...
            PosPre = 'START'
            PosPP = ''
            for line in csv.reader(ftrain,delimiter='\t'):
                if len(line) == 0:
                    Pos = 'END'
                    if (PosPP,PosPre) not in Ptrans:
                        Ptrans[(PosPP,PosPre)] = {Pos:1}
                    elif Pos not in Ptrans[(PosPP,PosPre)]:
                        Ptrans[(PosPP,PosPre)][Pos] = 1
                    else:
                        Ptrans[(PosPP,PosPre)][Pos] += 1
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
                if PosPre == 'START':
                    if Pos not in Ptrans['START']:
                        Ptrans['START'][Pos] = 1
                    else:
                        Ptrans['START'][Pos] += 1
                else:
                    if (PosPP,PosPre) not in Ptrans:
                        Ptrans[(PosPP,PosPre)] = { Pos : 1 } 
                    elif Pos not in Ptrans[(PosPP,PosPre)]:
                        Ptrans[(PosPP,PosPre)][Pos] = 1
                    else:
                        Ptrans[(PosPP,PosPre)][Pos] += 1
                PosPP = PosPre
                PosPre = Pos 
            ftrain.close()

        self.PosSize = len(self.Pemit)
        self.label = {Pos:enum for enum, Pos in enumerate(self.Pemit)}
        tmp = [(self.label[Pos],Pos) for Pos in self.label]
        tmp.sort()
        self.tag = [t[1] for t in tmp]
        self.tag.append('START')
        self.label.update({'END':self.PosSize,'START':self.PosSize})
        
        # Suffix model
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
            else:
                self.suffix.pop(suf)
        # Morphological model
        self.morph = np.zeros([self.morphCatNum,self.PosSize+1])
        for word in self.Words:
            cat = self.morphCat(word)
            for Pos in self.Words[word].keys():
                if cat > 1:
                    self.morph[cat,self.label[Pos]] += 1
                else:
                    if self.Words[word][Pos] <= threshold:
                        self.morph[cat,self.label[Pos]] += 1
        for vec in self.morph:
            vec = vec+1
            vec = 1./np.sum(vec)
        # Transition probabilities
        self.TransMat = np.zeros([self.PosSize+1,self.PosSize+1,self.PosSize+1])
        for PosPre in Ptrans:
            if PosPre == 'START':
                i = self.label['START']
                j = i
            else:
                i = self.label[PosPre[0]]
                j = self.label[PosPre[1]]
            for Pos in Ptrans[PosPre]:
                self.TransMat[i,j,self.label[Pos]] = Ptrans[PosPre][Pos] 
        self.TransMat2 = np.sum(self.TransMat,axis=0)
        self.TransMat += 1
        for mat in self.TransMat:
            for vec in mat:
                vec *= 1./np.sum(vec)
        for vec in self.TransMat2:
            vec *= 1./np.sum(vec)
        self.TransMat2 = self.TransMat2*np.ones([self.PosSize+1,self.PosSize+1,self.PosSize+1])
        # Emission rates
        self.unknown = np.zeros(self.PosSize+1)
        for Pos in self.Pemit:
            vec = np.array(self.Pemit[Pos].values())
            total = np.sum(vec)
            if Pos in self.openClass:
                self.unknown[self.label[Pos]] = np.sum(vec <= threshold)*1./total
            for word in self.Pemit[Pos]:
                self.Pemit[Pos][word] *= 1./total
        self.unknown *= 1./np.sum(self.unknown)
        # Get lambda 
        isg = self.TransMat[1:,:-1] >= self.TransMat2[1:,:-1]
        self.lam3 = np.sum(self.TransMat[1:,:-1][isg])
        self.lam2 = np.sum(self.TransMat2[1:,:-1][isg != True])
        total = self.lam2 + self.lam3
        self.lam3 /= total
        self.lam2 /= total
        # print self.lam3, self.lam2 

    # Generate the emission rate for unknown words
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

    # Solve the HMM using Viterbi Algorithm. 
    # Assume 2nd-Order Markov Model. 
    def tagSentence(self,snt):
        T = len(snt)
        Vtb = np.zeros([T+2,self.PosSize+1,self.PosSize+1])
        Trace = np.ones([T+2,self.PosSize+1,self.PosSize+1])*-1
        Vtb[0,self.label['START'],:] += 1
        # Vtb = np.log2(Vtb)
        ret = []
        for i in range(1,T+1):
            word = snt[i-1]
            PosSet = self.getPosTransEmit(word)
            for Pos,emit in PosSet:
                tmp = Vtb[i-1] * (self.lam3*self.TransMat[:,:,self.label[Pos]] + self.lam2*self.TransMat2[:,:,self.label[Pos]])
                Vtb[i,:,self.label[Pos]] = np.max(tmp,axis = 0) * emit *100
                Trace[i,:,self.label[Pos]] = np.argmax(tmp,axis=0)
        i = T+1
        Pos = 'END'
        tmp = Vtb[i-1] * (self.lam3*self.TransMat[:,:,self.label[Pos]] + self.lam2*self.TransMat2[:,:,self.label[Pos]]) 
        Vtb[i,:,self.label[Pos]] = np.max(tmp,axis = 0)
        Trace[i,:,self.label[Pos]] = np.argmax(tmp,axis=0)
        ToPos = self.label['END']
        FromPos = int(np.argmax(Vtb[i,:,ToPos]))
        PrePos = int(Trace[i,FromPos,ToPos])
        for i in range (T,0,-1):
            ret.append(self.tag[FromPos])
            ToPos = FromPos
            FromPos = PrePos
            PrePos = int(Trace[i,FromPos,ToPos])
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
    if len(sys.argv) <= 2:
        paths = ['WSJ_02-21.pos']
    if len(sys.argv) >= 2:
        testPath = sys.argv[1]
    if len(sys.argv) >= 3:
        paths = sys.argv[2:]
    tagger = POStagger_HMM()
    tagger.train(paths)
    if len(sys.argv) == 1:
        while True:
            try:
                sentence = input('Enter a sentence: ')
                snt = sentence.split(' ')
                try:
                    tag = tagger.tagSentence(snt)
                    for x,y in zip(sentence.split(' '),tag):
                        print x,'\t',y
                except:
                    pass
            except:
                break
    else: 
        fout = open(testPath+".pos","w")
        tagger.tagFile(dataPath+testPath+".words",fout)
        fout.close()
