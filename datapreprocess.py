import nltk

#access the data
def get_txt(filename):
    
    sample_letters = []
    conbined_sample_letters = []
    
    #read sample textfile(1~10) and text written by young for evaluation(cosine similarity)
    for i in range(1, 12):
        with open(f"{filename}_{i}.txt") as f:
            sample_letter = f.read()
            sample_letters.append(sample_letter)
            conbined_sample_letters.append(sample_letter)
    
    return sample_letters, conbined_sample_letters #return sample_letters list


#preprocess the data
def tokenize_txt(sample_letters):
    
    tokenized_sample_letters = []
    
    for sample_letter in sample_letters:
        tokenized_sample_letter = nltk.word_tokenize(sample_letter) #tokenize letter
        tokenized_sample_letters.append(tokenized_sample_letter)
    
    return tokenized_sample_letters #return tokenized_sample_letters list