from datapreprocess import get_txt
from datapreprocess import tokenize_txt
from analysis import count_words
from analysis import calc_tfidf
from analysis import cosine_similarity
import numpy as np

#word list

#https://www.verywellfamily.com/a-teen-slang-dictionary-2610994
young_words = ['dope', 'goat', 'gucci', 'lit', 'omg', 'salty', 'sic', 'sick', 'snatched', 
'fire', 'tbh', 'tea', 'thirsty', 'yolo', 'bae', 'basic', 'bf', 'gf', 'bff', 'bruh', 
'bro', 'dude', 'cap', 'curve', 'emo', 'fam', 'flex', 'karen', 'noob', 'periodt', 'ship', 
'shook', 'squad', 'sus', 'tight', 'tool', 'crashy', 'crunk', 'hangry', 
'requestion', 'tope', '53x', 'cu46', 'dayger', 'function', 'func', 'molly', 
'netflix', 'chill', 'rager', 'smash', 'sloshed', 'plug', 'turnt', 'x', 'wttp', 'lmirl', 'fuck',
'fucking', 'u', 'lol', 'ty', 'rn', 'wtf', 'mf', 'mb', 'savage', 'cu', 'U']


#https://www.dvusd.org/cms/lib/AZ01901092/Centricity/Domain/2891/SeniorVocabularyWords.pdf
#https://www.lifehack.org/articles/communication/24-old-english-terms-you-should-start-using-again.html
elderly_words = ['bedward', 'billingsgate', 'brabble', 'crapulous', 'elflock', 'erstwhile', 
'expergefactor', 'fudgel', 'groke', 'grubble', 'hugger-mugger', 'hum durgeon', 'jargogle', 
'lanspresado', 'mumpsimus', 'quagswag', 'rawgabbit', 'snottor', 'snollygoster', 'trumpery', 
'uhtceare', 'ultracrepidarian', 'zwodder', 'cockalorum', 'adroit', 'adulterate', 'adventitious', 
'aegis', 'asethetic', 'affectation', 'affinity', 'affluence', 'agape', 'agrandize', 'altruism', 
'ambiguous', 'amoral', 'amorphous', 'animosity', 'antipathy', 'antithesis', 'badinage', 'banal', 
'baroque', 'bauble', 'bedlam', 'beguile', 'besiege', 'besmirch', 'bestial', 'bilious', 'blanch', 
'bland', 'blandishment', 'bombast', 'boor', 'bovine', 'bowdlerize', 'brevity', 'bucolic', 
'cajole', 'callow', 'carcinogen', 'carnal', 'carrion', 'cataclysm', 'cataract', 'caveat', 'celibate', 
'censure', 'cessation', 'chaff', 'chagrin', 'chimerical', 'coalesce', 'debacle', 'debauchery', 'deference', 
'defile', 'deign', 'delineate', 'demeanor', 'denouement', 'deride', 'desiccated', 'despicable', 'desultory', 
'deviate', 'diadem', 'diaphanous', 'dichotomy', 'ebullient', 'eclectic', 'edify', 'dffete', 'egregious', 
'elegy', 'elicit', 'elixir', 'elucidata', 'emanate', 'emendation', 'empathy', 'empirical', 'endemic', 'enervate', 
'ennui', 'ephemeral', 'epitome', 'ergo', 'erotic', 'eschew', 'facetious', 'factious', 'fastidious', 'fatuous', 
'fecund', 'ferret', 'fervent', 'fervent', 'fetish', 'finesse', 'fiscal', 'fissure', 'flaccid', 'flagellate', 
'flaunt', 'flout', 'forment', 'fop', 'fortuitous', 'gambol', 'garish', 'garner', 'garrulous', 'germane', 
'gibe', 'gloat', 'glower', 'grandiose', 'gratuitous', 'grotesque', 'gumption', 'hackneyed', 'halcyon', 
'hallow', 'harbinger', 'harlequin', 'hector', 'hedonism', 'hegira', 'hermetic', 'heterogeneous', 
'hiatus', 'hoi polloi', 'hospice', 'hubrid', 'idiosyncrasy', 'idolatry', 'ignoble', 'imminent', 
'immolate', 'immutable', 'impair', 'impale', 'impalpable', 'impecunious', 'impediment', 'imperious', 
'impinge', 'impious', 'importune', 'impotent', 'imprecation', 'jocular', 'juxtapose', 'kinetic', 
'kismet', 'knell', 'labyrinth', 'lachrymose', 'laconic', 'lambent', 'languid', 'lascivious', 
'legerdemain', 'libertine', 'machination', 'macroscopic', 'maelstrom', 'malapropism', 'malleable', 
'martinet', 'masochist', 'mendacious', 'meretricious', 'milieu', 'miscreant', 'nebulous', 'necromancy', 
'neologism', 'nihilism', 'nirvana', 'nonentity', 'non sequitur', 'nubile', 'obdurate', 'obfuccate', 
'obloquy', 'obsequious', 'obviate', 'offal', 'olfactory', 'onerous', 'onus', 'optimum', 'opulent', 
'orifice', 'orthography', 'paleontology', 'palliate', 'panache', 'pandemic', 'panegyric', 'paradigm', 
'parochial', 'parody', 'paroxysm', 'peccadillo', 'pecuniary', 'pedantic', 'pedestrian', 'pejorative', 
'perdition', 'perfunctory', 'perspicacity', 'peruse', 'quagmire', 'quandary', 'quasi', 'querulous', 
'quiddity', 'raiment', 'rakish', 'ratiocinate', 'rationalize', 'rebuke', 'recant', 'recapitulate', 
'recoil', 'recondite', 'recreant', 'retify', 'redolent', 'redundant', 'regale', 'regress', 'sacrosanct', 
'sadistic', 'sagacious', 'salacious', 'salient', 'salutary', 'sangfroid', 'sanguine', 'savant', 
'scintillate', 'scurrilous', 'sedition', 'sedulous', 'sentient', 'shard', 'shibboleth', 'sibilant']

#print result of cosine similarity
def print_result1(TFIDFtable, conbined_sample_letters):

    #print cosine similarity sample letters and letters written by young people
    for i in range(len(conbined_sample_letters)):
        if i == 10:
            break
        a = np.array(TFIDFtable[i])
        b = np.array(TFIDFtable[10])
        print(f"sample {i+1} & letters written by young people(sample_11.txt) : {cosine_similarity(a, b)}")

#print the number of young words + cosine similarity sample letters and letters written by young people
def print_result2(TFIDFtable, conbined_sample_letters, count_young_words):

    for i in range(len(conbined_sample_letters)):
        if i == 10:
            break
        a = np.array(TFIDFtable[i])
        b = np.array(TFIDFtable[10])
        print(f"the number of young words + 'cosine similarity of sample {i+1} & letters written by young people(sample_11.txt)' : {count_young_words[i] + cosine_similarity(a, b)}")


#data access & preprocess sample letters
#sample_letters, conbined_sample_letters = get_txt("sample_src/sample")
sample_letters, conbined_sample_letters = get_txt("twitter/twitter")
tokenized_sample_letters = tokenize_txt(sample_letters)

#print the number of young words
print("the number of young words")
count_young_words = []
for i, tokenized_sample_letter in enumerate(tokenized_sample_letters):
    count = count_words(tokenized_sample_letter, young_words)
    print(f"sample {i+1} : {count}")
    count_young_words.append(count)
    
print()

#print the number of elderly words
print("the number of elderly words")
count_elderly_words = []
for i, tokenized_sample_letter in enumerate(tokenized_sample_letters):
    count = count_words(tokenized_sample_letter, elderly_words)
    print(f"sample {i+1} : {count}")
    count_elderly_words.append(count)

#calculate TF-IDF
TFIDFtable = calc_tfidf(conbined_sample_letters, conbined_sample_letters)

print()
print("cosine similarity")
print_result1(TFIDFtable, conbined_sample_letters)


#the number of young words + cosine similarity
print()
print("the number of young words + cosine similarity")
print_result2(TFIDFtable, conbined_sample_letters, count_young_words)