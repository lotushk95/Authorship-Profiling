from collections import Counter
from collections import defaultdict
import math
import numpy as np

#count the number word in letters and word list
def count_words(tokenized_words, vocabulary):
    
    count = 0
    
    #compare word
    for tokenized_word in tokenized_words:
        for word in vocabulary:
            if tokenized_word.lower() == word.lower():
                count += 1
    
    return count

#calculate tf-idf based on https://web-int.u-aizu.ac.jp/~paikic/lecture/Factory/material/exercise/ex-TFIDF/
def calc_tfidf(input_letters, conbined_sample_letters):
    N = len(input_letters)
    words = "".join(input_letters).split()
    count = Counter(words).most_common()
    
    #build dictionaries
    rdic = [i[0] for i in count]
    
    #calculating TFIDF
    letter_TFtable = defaultdict(Counter)
    letter_DFtable = Counter()
    letter_TFIDFtable = defaultdict(Counter)
    
    #calculate term frequency
    for letter in input_letters:
        words = letter.split()
        for word in words:
            letter_TFtable[letter][word] += 1
            
    #calculate document frequency
        for kw in letter_TFtable[letter].keys():
            letter_DFtable[kw] += 1
            
    #calculate TF-IDF
    for letter in input_letters:
        for kw in letter_TFtable[letter].keys():
            letter_TFIDFtable[letter][kw] = letter_TFtable[letter][kw] * math.log(N/letter_DFtable[kw])
            
          
    # make vector for calculating cosine similarity  
    TFIDFtable = [[0.0 for _ in range(len(rdic))] for _ in range(len(letter_TFIDFtable))]

    for i in range(len(conbined_sample_letters)):
        for j in range(len(rdic)):
            keys_array = letter_TFIDFtable[conbined_sample_letters[i]].keys()
            for key in keys_array:
                if rdic[j] == key:
                    TFIDFtable[i][j] = letter_TFIDFtable[conbined_sample_letters[i]][key]
    
    return TFIDFtable #return vector for cosine similarity

#calculate cosine similarity of sample letters
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    cos_sim = dot_product / (norm_a * norm_b)
    
    return cos_sim #return cosine similarity