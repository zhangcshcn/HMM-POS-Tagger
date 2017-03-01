Author: Chen Zhang, with NYU Courant Institute of Mathematical Science  
Email:  chen.zhang@nyu.edu

### Brief Description

This is the solution to assignment 4 of CSCI_GA.2590 Natural Language Processing Spring 2017, taught by Prof. Grishman. The code carries out Part-of-Speech tagging using HMM model. 

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
> 1. python tagPOS_hmm.py  
> 
   This will take *WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos* as the training file,  
   and let the user enter space divided tokens with sentence structure as input.  
> 
> 2. python tagPOS_hmm.py testfile  
> 
    This will take *WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos* as the training file,  
    and use  *WSJ_POS_CORPUS_FOR_STUDENTS/testfile.words* as test input.  
    *./testfile.pos* will be generated as output.   
> 
> 3. python tagPOS_hmm.py testfile training-file1.pos training-file2.pos ...  
> 
    This will take *WSJ_POS_CORPUS_FOR_STUDENTS/training-file1.pos*, *WSJ_POS_CORPUS_FOR_STUDENTS/training-file2.pos*, ... as the training files,  
    and use  *WSJ_POS_CORPUS_FOR_STUDENTS/testfile.words* as test input.  
    *./testfile.pos* will be generated as output.  

