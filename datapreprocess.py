import nltk

#access the data
def get_txt(filename):
    
    sample_letters = []
    
    for i in range(1, 11):
        with open(f"{filename}_{i}.txt") as f:
            sample_letter = f.read()
            sample_letters.append(sample_letter)
    
    return sample_letters


#preprocess the data
def tokenize_txt(sample_letters):
    
    tokenized_sample_letters = []
    
    for sample_letter in sample_letters:
        tokenized_sample_letter = nltk.word_tokenize(sample_letter)
        tokenized_sample_letters.append(tokenized_sample_letter)
    
    return tokenized_sample_letters

#test code
'''
sample_letters = get_txt("sample_src/sample")
tokenizes_sample_letters = tokenize_txt(sample_letters)
print(count_words(tokenizes_sample_letters[], young_words))
'''