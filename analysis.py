from collections import Counter
from collections import defaultdict
import math
import numpy as np

#count the number word in letters and word list
def count_words(tokenized_words, vocabulary):
    
    count = 0
    
    for tokenized_word in tokenized_words:
        for word in vocabulary:
            if tokenized_word.lower() == word.lower():
                #print(tokenized_word)
                count += 1
    
    return count

#calculate tf-idf
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
    
    for letter in input_letters:
        words = letter.split()
        for word in words:
            letter_TFtable[letter][word] += 1

        for kw in letter_TFtable[letter].keys():
            letter_DFtable[kw] += 1

    for letter in input_letters:
        for kw in letter_TFtable[letter].keys():
            letter_TFIDFtable[letter][kw] = letter_TFtable[letter][kw] * math.log(N/letter_DFtable[kw])

    TFIDFtable = [[0.0 for _ in range(len(rdic))] for _ in range(len(letter_TFIDFtable))]

    for i in range(len(conbined_sample_letters)):
        for j in range(len(rdic)):
            keys_array = letter_TFIDFtable[conbined_sample_letters[i]].keys()
            for key in keys_array:
                if rdic[j] == key:
                    TFIDFtable[i][j] = letter_TFIDFtable[conbined_sample_letters[i]][key]
    
    return TFIDFtable


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    cos_sim = dot_product / (norm_a * norm_b)
    
    return cos_sim

def print_result(TFIDFtable, conbined_sample_letters):

    for i in range(len(conbined_sample_letters)):
        for j in range(len(conbined_sample_letters)):
            a = np.array(TFIDFtable[i])
            b = np.array(TFIDFtable[j])
            print(f"{i+1}:{j+1}")
            print((cosine_similarity(a, b)))