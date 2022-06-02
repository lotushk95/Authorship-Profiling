import nltk
from datapreprocess import get_txt
from datapreprocess import tokenize_txt


#counting the number word in letters and word list
def count_words(tokenized_words, vocabulary):
    
    count = 0
    
    for tokenized_word in tokenized_words:
        for word in vocabulary:
            if tokenized_word.lower() == word.lower():
                print(tokenized_word)
                count += 1
    
    return count


