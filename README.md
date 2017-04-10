Author: Chen Zhang, with NYU Courant Institute of Mathematical Science  
Email:  chen.zhang@nyu.edu

### Brief Description

This is the solution to assignment 4 of CSCI_GA.2590 Natural Language Processing Spring 2017, taught by Prof. Grishman. The code carries out Part-of-Speech tagging using HMM model.  
This every submission got the *BEST performance* among all my peers.  
很惭愧，只做了一点微小的贡献。

### Features

* 1st-Order Markov Model and 2nd-Order Markov model are hybrided in the solution. 
* Unknown words handled using
    - Hapax Legomena with open class to approximate the basic POS distribution over unknown word 
    - morphological features 
        - 16 catagories, with regard to Capitalizations, hyphens, digits, and symbols.
        - each word counts only once
    - suffix features 
        - 2-letter suffices with adequate number of times occurance 
        - each word conuts only once
* Vectorized and Matricized computation for better performance 

### Requiremnts

* numpy

### Instrucions
Place the *tagPOS_hmm.py* in the parent directory of *WSJ_POS_CORPUS_FOR_STUDENTS*  

There are 3 ways to run the tagger: 
> 
> python tagPOS_hmm.py  
> 
   This will take *WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos* as the training file,  
   and let the user enter space divided tokens with sentence structure as input.  
> 
> python tagPOS_hmm.py testfile  
> 
    This will take *WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos* as the training file,  
    and use  *WSJ_POS_CORPUS_FOR_STUDENTS/testfile.words* as test input.  
    *./testfile.pos* will be generated as output.   
> 
> python tagPOS_hmm.py testfile training-file1.pos training-file2.pos ...  
> 
    This will take 
    
    - *WSJ_POS_CORPUS_FOR_STUDENTS/training-file1.pos*, 
    - *WSJ_POS_CORPUS_FOR_STUDENTS/training-file2.pos*, 
    - ... 
    
    as the training files,  
    and use  
    
    - *WSJ_POS_CORPUS_FOR_STUDENTS/testfile.words* 
    
    as test input.  
    
    - *./testfile.pos* 
    
    will be generated as output.  

### Reference
1. Thorsten Brants. 2000. TnT: a statistical part-of-speech tagger. In Proceedings of the sixth conference on Applied natural language processing (ANLC '00). Association for Computational Linguistics, Stroudsburg, PA, USA, 224-231. DOI=http://dx.doi.org/10.3115/974147.974178 
2. Ralph Weischedel, Richard Schwartz, Jeff Palmucci, Marie Meteer, and Lance Ramshaw. 1993. Coping with ambiguity and unknown words through probabilistic models. Comput. Linguist. 19, 2 (June 1993), 361-382. 
3. Ratnaparkhi, A., 1996. A maximum entropy model for part-of-speech tagging. In Proceedings of the conference on empirical methods in natural language processing (Vol. 1, pp. 133-142).

