
Katherine Kairis, kak275@pitt.edu, 12/15/2017

# Hesitation Analysis
## Table of Contents
* [Helper functions](#helper-functions)
* [Repeated words/stuttering](#repeated-words/stuttering)
* [Proportion of Hesitation words (er, erm, etc.)](#proportion-of-hesitation-words-(er,-erm,-etc.))


```python
import pickle
import nltk
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
#Get the tokens for each corpus from their respective pickle files
f = open('VOICE_tokenized.p', 'rb')
VOICE_toks = pickle.load(f)
f.close()

f = open('BNC_tokenized.p', 'rb')
BNC_toks = pickle.load(f)
f.close()

f = open('VOICE_tagged.p', 'rb')
VOICE_tags = pickle.load(f)
f.close()

f = open('BNC_tagged.p', 'rb')
BNC_tags = pickle.load(f)
f.close()
```

## Helper functions
* get_tokens
* get_bigrams
* repeated_words
* repeated_stopwords
* tag_counts


```python
def get_tokens(dictionary):
    tokens = []
    for file in dictionary:
        for key in dictionary[file]:
            tokens.extend(dictionary[file][key])
    return tokens
```


```python
def get_bigrams(dictionary):
    bigrams = []
    for file in dictionary:
        for key in dictionary[file]:
            pairs = list(nltk.bigrams(dictionary[file][key]))
            bigrams.extend(pairs)
    return bigrams
```


```python
"""
Takes a list of bigrams, returns dictionary whose keys are bigrams containing duplicate words (e.g ('i', 'i',), 
'the', 'the') and whose values are the frequencies of each bigram
"""
def repeated_words(bigrams):
    repeated = {}
    for b in bigrams:
        if(b[0] == b[1]):
            if(b not in repeated):
                repeated[b] = 1
            else:
                repeated[b] += 1
    return repeated
```




    "\nTakes a list of bigrams, returns dictionary whose keys are bigrams containing duplicate words (e.g ('i', 'i',), \n'the', 'the') and whose values are the frequencies of each bigram\n"




```python
"""
Takes a dictionary of bigrams whose keys are bigrams containing duplicate words (e.g ('i', 'i',), 
'the', 'the') and whose values are the frequencies of each bigram. Returns the sum of the frequencies for only
bigrams that contain repeated stop words
"""
def repeated_stopwords(bigram_dict):
    count = 0
    for b in bigram_dict.keys():
        if b[0] in stopWords:
            count += bigram_dict[b]
    return count
```




    "\nTakes a dictionary of bigrams whose keys are bigrams containing duplicate words (e.g ('i', 'i',), \n'the', 'the') and whose values are the frequencies of each bigram. Returns the sum of the frequencies for only\nbigrams that contain repeated stop words\n"




```python
"""
Takes a dictionary containing conversations and a user-provided tag. Returns a dictionary who keys are words that are 
associated with the tag, and whose values are the frequencies of each word
"""
def tag_counts(conv_dictionary, tag):
    tags = {}
    for file in conv_dictionary:
        for key in conv_dictionary[file]:
            for pair in conv_dictionary[file][key]:
                if(pair[1] == tag):
                    if(pair not in tags):
                        tags[pair] = 1
                    else:
                        tags[pair] += 1
    return tags
```




    '\nTakes a dictionary containing conversations and a user-provided tag. Returns a dictionary who keys are words that are \nassociated with the tag, and whose values are the frequencies of each word\n'



## Repeated words/stuttering


```python
#Get the bigrams from VOICE and BNC
VOICE_bigrams = get_bigrams(VOICE_toks)
BNC_bigrams = get_bigrams(BNC_toks)

len(VOICE_bigrams)
len(BNC_toks)
```
    541521
    908




```python
#Get the bigrams with repeated words, like ('i', 'i'), and their frequency counts
VOICE_repeated_words = repeated_words(VOICE_bigrams)
BNC_repeated_words = repeated_words(BNC_bigrams)
```

### Proportion of bigrams that contain repeated words
When comparing VOICE to BNC, the speakers in the VOICE corpus repeat words significantly more often than the speakers in the BNC. Around 3% of all bigrams in the VOICE corpus contain repeated words, while only around 1% of all bigrams in the BNC contain repeated words


```python
#For each corpus, get the sum of the frequencies for each bigram containing a repeated word
VOICE_total_repetitions =  sum(VOICE_repeated_words.values())
BNC_total_repetitions = sum(BNC_repeated_words.values())
```


```python
#The VOICE corpus contains 15819 bigrams with repeated words, which accounts for almost 3% of all of the bigrams
#in VOICE
VOICE_total_repetitions
VOICE_total_repetitions/len(VOICE_bigrams)
```
    15819
    0.02921216351720432




```python
#The BNC corpus contains 101555 bigrams with repeated words, which accounts for around 1% of all bigrams in the BNC
BNC_total_repetitions
BNC_total_repetitions/len(BNC_bigrams)
```
    101555
    0.010828264398376917



### Exploring words that are often repeated
There are a lot of similarities between the most-commonly repeated words for both native and nonnative speakers ('yeah,' 'er,' 'I'), but there are also potentially some important differences. For example, 'very very' is a common bigram in BNC. English speakers often intentionally repeat the word "very" for emphasis (not hesitation). This could be the case for some of the bigrams with repeated words in the BNC.  
Also, a high proportion of repeated words (60%+) are stop words. This proportion is higher among native speakers in the BNC than it is among non-native speakers in VOICE.


```python
#Show the 20 most frequently repeated words in VOICE
for bigram in sorted(VOICE_repeated_words, key = VOICE_repeated_words.get, reverse=True)[:20]:
    print(bigram, VOICE_repeated_words[bigram])
```

    ('yeah', 'yeah') 1241
    ('the', 'the') 1177
    ('er', 'er') 1096
    ('i', 'i') 1023
    ('no', 'no') 846
    ('to', 'to') 672
    ('in', 'in') 615
    ('mhm', 'mhm') 556
    ('and', 'and') 510
    ('yes', 'yes') 476
    ('we', 'we') 433
    ('yah', 'yah') 380
    ('x', 'x') 352
    ('a', 'a') 311
    ('it', 'it') 293
    ('you', 'you') 288
    ('of', 'of') 275
    ('that', 'that') 213
    ('xx', 'xx') 186
    ('for', 'for') 180



```python
#Show the 20 most frequently repeated words in BNC
for bigram in sorted(BNC_repeated_words, key=BNC_repeated_words.get, reverse=True)[:20]:
    print(bigram, BNC_repeated_words[bigram])
```

    ('i', 'i') 8598
    ('the', 'the') 6722
    ('that', 'that') 3953
    ('no', 'no') 3948
    ('and', 'and') 3511
    ('er', 'er') 3448
    ('a', 'a') 3228
    ('mm', 'mm') 3197
    ('it', 'it') 2940
    ('you', 'you') 2922
    ('in', 'in') 2486
    ('yeah', 'yeah') 2248
    ('to', 'to') 2015
    ('we', 'we') 2004
    ('yes', 'yes') 1861
    ('they', 'they') 1742
    ('he', 'he') 1670
    ('is', 'is') 1571
    ('very', 'very') 1437
    ('what', 'what') 1342


### Repetition and stop words
In both the BNC and VOICE, over 60% of repeated words are stop words. This percentage is higher in the BNC than in the VOICE (66.75% vs. 60.3%)


```python
#For each corpus, get the number of repeated words that are stop words
VOICE_repeated_stopwords = repeated_stopwords(VOICE_repeated_words)
BNC_repeated_stopwords = repeated_stopwords(BNC_repeated_words)
```


```python
#For each corpus, display the proportion of repeated words that are stop words
VOICE_repeated_stopwords/VOICE_total_repetitions
BNC_repeated_stopwords/BNC_total_repetitions
```
    0.6030090397623111
    0.6675889911870415



#### Considering non-repeated stop words


```python
VOICE_repeated_non_stopwords = {}
for b in VOICE_repeated_words:
    if b[0] not in stopWords:
        if b not in VOICE_repeated_non_stopwords:
            VOICE_repeated_non_stopwords[b] = VOICE_repeated_words[b]
```


```python
BNC_repeated_non_stopwords = {}
for b in BNC_repeated_words:
    if b[0] not in stopWords:
        if b not in BNC_repeated_non_stopwords:
            #BNC_repeated_non_stopwords[b] = 1
            BNC_repeated_non_stopwords[b] = BNC_repeated_words[b]
```


```python
for bigram in sorted(VOICE_repeated_non_stopwords, key= VOICE_repeated_non_stopwords.get, reverse=True)[:20]:
    print(bigram, VOICE_repeated_non_stopwords[bigram])
```

    ('yeah', 'yeah') 1241
    ('er', 'er') 1096
    ('mhm', 'mhm') 556
    ('yes', 'yes') 476
    ('yah', 'yah') 380
    ('x', 'x') 352
    ('xx', 'xx') 186
    ('okay', 'okay') 134
    ('erm', 'erm') 116
    ('hm', 'hm') 75
    ('right', 'right') 73
    ('i-', 'i-') 60
    ('blah', 'blah') 41
    ('one', 'one') 37
    ('hh', 'hh') 35
    ('th-', 'th-') 32
    ('really', 'really') 32
    ('sure', 'sure') 32
    ('s-', 's-') 31
    ('like', 'like') 28



```python
for bigram in sorted(BNC_repeated_non_stopwords, key= BNC_repeated_non_stopwords.get, reverse=True)[:20]:
    print(bigram, BNC_repeated_non_stopwords[bigram])
```

    ('er', 'er') 3448
    ('mm', 'mm') 3197
    ('yeah', 'yeah') 2248
    ('yes', 'yes') 1861
    ('oh', 'oh') 820
    ('ha', 'ha') 751
    ('ah', 'ah') 563
    ('da', 'da') 535
    ('erm', 'erm') 519
    ('one', 'one') 498
    ('la', 'la') 434
    ('th', 'th') 414
    ('ooh', 'ooh') 398
    ('well', 'well') 380
    ('two', 'two') 328
    ('bye', 'bye') 328
    ('b', 'b') 318
    ('p', 'p') 307
    ('c', 'c') 275
    ('doo', 'doo') 257



```python
freqs = nltk.FreqDist([b for b in VOICE_repeated_words if b[0] not in stopWords])
```

## Proportion of Hesitation words (er, erm, etc.)
Both corpora have "unclassifiable" categories that hesitations fall under, but the criteria for each corpus are very different. VOICE includes mainly interjections and hesitation words, while BNC also includes words that were cut-off. However, the most common words in each corpus with the "unclassifiable" tags by far are 'er' and 'erm,' so I decided to create a set of "hesitation" words that is a combination of unclassifiable words in the BNC and the VOICE. Specifically, it includes words that are unclassifiable in the VOICE that are also unclassifiable in the BNC.  
There is a huge difference in the percent of hesitation words between the two corpora: about 43% of the words in VOICE are "hesitation" words, while about 24% of words in the BNC are "hesitation" words.


```python
"""
Both VOICE and BNC have a tag that is used for words that do not have another clear classification. Both corpora point
out that hesitation words are among the most common to fall into this category. In VOICE, they are indicated with the 
UHfUH tag, and in BNC, they are indicated with the UNC tag.
"""
#VOICE: UHfUH
#BNC: UNC
VOICE_hesitations = tag_counts(VOICE_tags, "UHfUH")
BNC_hesitations = tag_counts(BNC_tags, "UNC")
```






```python
#A LOT more words in the BNC fall under the unclassified category than in the BNC. This is because of the difference
#in tagsets
len(VOICE_hesitations.keys())
len(BNC_hesitations.keys())
```
    29
    5082




```python
#Get the number of words in VOICE with the UHfUH tag, and the number of words in BNC with the UNC  tag. 
VOICE_num_hesitations = sum(VOICE_hesitations.values())
BNC_num_hesitations = sum(BNC_hesitations.values())

VOICE_num_hesitations
BNC_num_hesitations
```

    43001
    231847




```python
#The tagging criteria for unclassified words between the BNC and VOICE are very different: the BNC applies this 
#tag to a much wider variety of words. For example, the BNC counts cut off words ('th-') as unclassified, while
#the VOICE does not.
list(VOICE_hesitations.keys())
list(BNC_hesitations.keys())[:30]
```
    [('er', 'UHfUH'),
     ('oh', 'UHfUH'),
     ('ah', 'UHfUH'),
     ('erm', 'UHfUH'),
     ('pf', 'UHfUH'),
     ('oops', 'UHfUH'),
     ('haeh', 'UHfUH'),
     ('wow', 'UHfUH'),
     ('sh', 'UHfUH'),
     ('ooph', 'UHfUH'),
     ('ur', 'UHfUH'),
     ('yo', 'UHfUH'),
     ('whoohoo', 'UHfUH'),
     ('yuck', 'UHfUH'),
     ('huh', 'UHfUH'),
     ('oh-oh', 'UHfUH'),
     ('poah', 'UHfUH'),
     ('ow', 'UHfUH'),
     ('ts', 'UHfUH'),
     ('oow', 'UHfUH'),
     ('innit', 'UHfUH'),
     ('yipee', 'UHfUH'),
     ('mm', 'UHfUH'),
     ('ha', 'UHfUH'),
     ('yay', 'UHfUH'),
     ('uh', 'UHfUH'),
     ('ouch', 'UHfUH'),
     ('psh', 'UHfUH'),
     ('eh', 'UHfUH')]

    [('erm', 'UNC'),
     ('er', 'UNC'),
     ("'s", 'UNC'),
     ('be', 'UNC'),
     ('th-', 'UNC'),
     ('com', 'UNC'),
     ('gu', 'UNC'),
     ('di', 'UNC'),
     ('creme', 'UNC'),
     ('non', 'UNC'),
     ('lieu', 'UNC'),
     ('en', 'UNC'),
     ('int', 'UNC'),
     ('ma', 'UNC'),
     ('te', 'UNC'),
     ('cur', 'UNC'),
     ('in', 'UNC'),
     ('s', 'UNC'),
     ('pa', 'UNC'),
     ('si', 'UNC'),
     ('wh', 'UNC'),
     ('ac', 'UNC'),
     ('thi', 'UNC'),
     ('st', 'UNC'),
     ("'", 'UNC'),
     ('papier', 'UNC'),
     ('mache', 'UNC'),
     ('mark', 'UNC'),
     ('p', 'UNC'),
     ('walk', 'UNC')]




```python
#Make a list of hesitation words, which will consist of words in the VOICE marked as unclassified that are
#also marked as unclassified in the BNC

VOICE_hesitation_words = list(VOICE_hesitations.keys())
VOICE_hesitation_words = [w[0] for w in VOICE_hesitation_words]

BNC_hesitation_words = list(VOICE_hesitations.keys())
BNC_hesitation_words = [w[0] for w in BNC_hesitation_words]

hesitation_words = [w for w in VOICE_hesitation_words if w in BNC_hesitation_words]
hesitation_words
```
    ['er',
     'oh',
     'ah',
     'erm',
     'pf',
     'oops',
     'haeh',
     'wow',
     'sh',
     'ooph',
     'ur',
     'yo',
     'whoohoo',
     'yuck',
     'huh',
     'oh-oh',
     'poah',
     'ow',
     'ts',
     'oow',
     'innit',
     'yipee',
     'mm',
     'ha',
     'yay',
     'uh',
     'ouch',
     'psh',
     'eh']




```python
BNC_tokens = get_tokens(BNC_toks)
VOICE_tokens = get_tokens(VOICE_toks)
```


```python
#Huge difference in hesitation words

len([w for w in VOICE_tokens if w in hesitation_words])/len(VOICE_tokens)
len([w for w in BNC_tokens if w in hesitation_words])/len(BNC_tokens)
```
    0.0428130167749006
    0.024098311427494978


