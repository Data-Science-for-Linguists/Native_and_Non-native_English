
Katherine Kairis, kak275@pitt.edu, 12/15/2017

# Analyzing Specific L1s


```python
import pickle
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
stopWords = set(stopwords.words('english'))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
f = open('VOICE_tokenized.p', 'rb')
VOICE_toks = pickle.load(f)
f.close()

f = open('VOICE_tagged.p', 'rb')
VOICE_tags = pickle.load(f)
f.close()

f = open('VOICE_native_tagged.p', 'rb')
VOICE_native_tags = pickle.load(f)
f.close()

f = open('VOICE_participant_info.p', 'rb')
participants = pickle.load(f)
f.close()
```

## VOICE Tagset
"CC" coordinating conjunction
"CD" cardinal number  
"DM" discourse marker  
"DT" determiner 
"EX" existential there 
"FI" formulaic items 
"FW" foreign word (non-English speech) 
"IN" preposition or subordinating conjunction  
"JJ" adjective, positive
"JJR"adjective, comparative 
"JJS"adjective, superlative 
"LA" laughter
"MD" modal verb  
"NN" noun, singular or mass
"NNS" noun, plural 
"NP" proper noun, singular
"NPS" proper noun, plural  
"PA" pause  
"PP" pronoun, personal  
"PPS" pronoun, possessive 
"PRE" pronoun, relative 
"RB" adverb, positive
"RBR" adverb, comparative 
"RE" response particle  
"RP" particle 
"TO" to, infinitive use  
"UH" interjection
"UNI" unintelligible speech 
"V" generic verb tag 
"VB verb be, base form 
"VBD" verb be, past tense
"VBN" verb be, past participle 
"VBP" verb be, present non-3rd person singular  
"VBS" verb be, contracted form 
VBZ" verb be, present 3rd person singular 
"VH" verb have, base form
"VHG" verb have, gerund or present participle
"VHP" verb have, present non-3rd person singular  
"VHS" verb have, contracted form  
"VV" other verbs than be and have, base form 
"VVD" other verbs than be and have, past tense 
"VVG"other verbs than be and have, gerund or present participle 
"VVN" other verbs than be and have, past participle  
"VVP" other verbs than be and have, present non-3rd person singular  
"VVZ" other verbs than be and have, present 3rd person singular  
"WDT" wh-determiner 
"WP" wh-pronoun  
"WRB" wh-adverb 
"XX" partial word 

## Getting speakers of chosen languages
Non-native L1s selected: Polish, Korean, Finnish, Danish, Turkish, and Portuguese  
Will compare all non-native L1s to English


```python
def get_monolingual_speakers(participant_dict, language):
    speakers = []
    for p in participant_dict.keys():
        L1s = participant_dict[p]['L1']
        if(len(L1s) == 1 and language in L1s):
            speakers.append(p)
            
    return speakers
```


```python
eng_speakers = get_monolingual_speakers(participants, 'eng')
pol_speakers = get_monolingual_speakers(participants, 'pol')
kor_speakers = get_monolingual_speakers(participants, 'kor')
fin_speakers = get_monolingual_speakers(participants, 'fin')
dan_speakers = get_monolingual_speakers(participants, 'dan')
tur_speakers = get_monolingual_speakers(participants, 'tur')
hun_speakers = get_monolingual_speakers(participants, 'hun')
por_speakers = get_monolingual_speakers(participants, 'por')
rus_speakers = get_monolingual_speakers(participants, 'rus')
mlt_speakers = get_monolingual_speakers(participants, 'mlt')
lav_speakers = get_monolingual_speakers(participants, 'lav')
```


```python
print("English speakers:", len(eng_speakers))
print("Polish speakers:", len(pol_speakers))
print("Korean speakers:", len(kor_speakers))
print("Finnish speakers:", len(fin_speakers))
print("Danish speakers:", len(dan_speakers))
print("Turkish speakers:", len(tur_speakers))
print("Hungarian speakers:", len(hun_speakers))
print("Portuguese speakers:", len(por_speakers))
print("Russian speakers:", len(rus_speakers))
print("Maltese speakers:", len(mlt_speakers))
print("Latvian speakers:", len(lav_speakers))
```

    English speakers: 62
    Polish speakers: 35
    Korean speakers: 14
    Finnish speakers: 51
    Danish speakers: 35
    Turkish speakers: 14
    Hungarian speakers: 13
    Portuguese speakers: 21
    Russian speakers: 22
    Maltese speakers: 22
    Latvian speakers: 19



```python
#Get the tokens for each speaker listed in the "speakers" list
def get_tagged_utterances(tokens, speakers): 
    utterances = []
    for conversation in tokens.keys():
        for pair in tokens[conversation]:
            if(pair[1] in speakers):
                utterances.append(tokens[conversation][pair])
                
    return utterances
```


```python
#Get the tokens from the speech of the English, Korean, Finnish, Turkish, Danish, Portuguese, and Polish speakers
eng_speech = get_tagged_utterances(VOICE_native_tags, eng_speakers)
kor_speech = get_tagged_utterances(VOICE_tags, kor_speakers)
fin_speech = get_tagged_utterances(VOICE_tags, fin_speakers)
tur_speech = get_tagged_utterances(VOICE_tags, tur_speakers)
dan_speech = get_tagged_utterances(VOICE_tags, dan_speakers)
por_speech = get_tagged_utterances(VOICE_tags, por_speakers)
pol_speech = get_tagged_utterances(VOICE_tags, pol_speakers)
```

## Trigrams


```python
def get_trigrams(li):
    trigram_list = []
    for trigram in list(nltk.trigrams(li)):
        if len(trigram) > 0:
            trigram_list.append(trigram)
            
    return trigram_list
        
```


```python
"""
Some tags to get rid of: BR(breathing), PA(pause), UH(interjections and hesitations), UNI(unintelligible), UNK(unknown)
https://www.univie.ac.at/voice/page/documents/VOICE_tagging_manual.pdf
"""   
def remove_tags(li):
    unwanted_tags = ["BRfBR", "PAfPA", "UHfUH", "UNIfUNI", "UNKfNN", "LAfLA", "XXfXX"]
    return [pair for pair in li if pair[1] not in unwanted_tags]
```




    '\nSome tags to get rid of: BR(breathing), PA(pause), UH(interjections and hesitations), UNI(unintelligible), UNK(unknown)\nhttps://www.univie.ac.at/voice/page/documents/VOICE_tagging_manual.pdf\n'




```python
#Returns a list of (word, tag) pairs
def get_pairs(li):
    tokens = []
    for u in li:
        for w in u:
            tokens.append(w)
    return tokens
```


```python
def get_tags(li):
    return[pair[1] for pair in li]
```


```python
#Get the tokens for each L1 group
eng_toks = get_pairs(eng_speech)
eng_toks = remove_tags(eng_toks)
eng_tags = get_tags(eng_toks)

kor_toks = get_pairs(kor_speech)
kor_toks = remove_tags(kor_toks)
kor_tags = get_tags(kor_toks)

fin_toks = get_pairs(fin_speech)
fin_toks = remove_tags(fin_toks)
fin_tags = get_tags(fin_toks)

tur_toks = get_pairs(tur_speech)
tur_toks = remove_tags(tur_toks)
tur_tags = get_tags(tur_toks)

dan_toks = get_pairs(dan_speech)
dan_toks = remove_tags(dan_toks)
dan_tags = get_tags(dan_toks)

por_toks = get_pairs(por_speech)
por_toks = remove_tags(por_toks)
por_tags = get_tags(por_toks)

pol_toks = get_pairs(pol_speech)
pol_toks = remove_tags(pol_toks)
pol_tags = get_tags(pol_toks)
```


```python
#Get the part of speech trigrams for each L1 group
eng_tag_trigrams = get_trigrams(eng_tags)
kor_tag_trigrams = get_trigrams(kor_tags)
fin_tag_trigrams = get_trigrams(fin_tags)
tur_tag_trigrams = get_trigrams(tur_tags)
dan_tag_trigrams = get_trigrams(dan_tags)
por_tag_trigrams = get_trigrams(por_tags)
pol_tag_trigrams = get_trigrams(pol_tags)
```

### Most frequent part-of-speech trigrams for each L1 group


```python
#Create frequency dictionaries containing the most common part of speech trigrams for each L1
eng_freq = nltk.FreqDist(eng_tag_trigrams)
kor_freq = nltk.FreqDist(kor_tag_trigrams)
fin_freq = nltk.FreqDist(fin_tag_trigrams)
tur_freq = nltk.FreqDist(tur_tag_trigrams)
dan_freq = nltk.FreqDist(dan_tag_trigrams)
por_freq = nltk.FreqDist(por_tag_trigrams)
pol_freq = nltk.FreqDist(pol_tag_trigrams)
```


```python
#Create a data frame containing each L1's 20 most common part of speech trigram
trigram_df = pd.DataFrame(
    {"L1=English": [t[0] for t in eng_freq.most_common(20)],
     "L1=Korean": [t[0] for t in kor_freq.most_common(20)],
     "L1=Finnish": [t[0] for t in fin_freq.most_common(20)],
     "L1=Turkish": [t[0] for t in tur_freq.most_common(20)],
     "L1=Danish": [t[0] for t in dan_freq.most_common(20)],
     "L1=Portuguese": [t[0] for t in por_freq.most_common(20)],
     "L1=Polish": [t[0] for t in pol_freq.most_common(20)]
    }
)

trigram_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L1=Danish</th>
      <th>L1=English</th>
      <th>L1=Finnish</th>
      <th>L1=Korean</th>
      <th>L1=Polish</th>
      <th>L1=Portuguese</th>
      <th>L1=Turkish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(INfIN, DTfDT, NNfNN)</td>
      <td>(INfIN, DTfDT, NNfNN)</td>
      <td>(INfIN, DTfDT, NNfNN)</td>
      <td>(REfRE, REfRE, REfRE)</td>
      <td>(INfIN, DTfDT, NNfNN)</td>
      <td>(INfIN, DTfDT, NNfNN)</td>
      <td>(INfIN, DTfDT, NNfNN)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(DTfDT, NNfNN, INfIN)</td>
      <td>(DTfDT, NNfNN, INfIN)</td>
      <td>(DTfDT, NNfNN, INfIN)</td>
      <td>(INfIN, DTfDT, NNfNN)</td>
      <td>(REfRE, REfRE, REfRE)</td>
      <td>(DTfDT, NNfNN, INfIN)</td>
      <td>(DTfDT, NNfNN, INfIN)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(DTfDT, JJfJJ, NNfNN)</td>
      <td>(DTfDT, JJfJJ, NNfNN)</td>
      <td>(REfRE, REfRE, REfRE)</td>
      <td>(DTfDT, JJfJJ, NNfNN)</td>
      <td>(DTfDT, NNfNN, INfIN)</td>
      <td>(NNfNN, INfIN, DTfDT)</td>
      <td>(DTfDT, JJfJJ, NNfNN)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(REfRE, REfRE, REfRE)</td>
      <td>(NNfNN, INfIN, DTfDT)</td>
      <td>(DTfDT, JJfJJ, NNfNN)</td>
      <td>(DTfDT, NNfNN, INfIN)</td>
      <td>(NNfNN, INfIN, DTfDT)</td>
      <td>(DTfDT, JJfJJ, NNfNN)</td>
      <td>(NNfNN, INfIN, DTfDT)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(NNfNN, INfIN, DTfDT)</td>
      <td>(REfRE, REfRE, REfRE)</td>
      <td>(NNfNN, INfIN, DTfDT)</td>
      <td>(NNfNN, INfIN, DTfDT)</td>
      <td>(DTfDT, JJfJJ, NNfNN)</td>
      <td>(PPfPP, MDfMD, VVfVV)</td>
      <td>(INfIN, DTfDT, JJfJJ)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(INfIN, DTfDT, JJfJJ)</td>
      <td>(PPfPP, MDfMD, VVfVV)</td>
      <td>(INfIN, DTfDT, JJfJJ)</td>
      <td>(FWfFW, FWfFW, FWfFW)</td>
      <td>(PPfPP, MDfMD, VVfVV)</td>
      <td>(INfIN, DTfDT, JJfJJ)</td>
      <td>(JJfJJ, NNfNN, INfIN)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(JJfJJ, NNfNN, INfIN)</td>
      <td>(INfIN, DTfDT, JJfJJ)</td>
      <td>(PPfPP, MDfMD, VVfVV)</td>
      <td>(PPfPP, MDfMD, VVfVV)</td>
      <td>(INfIN, DTfDT, JJfJJ)</td>
      <td>(TOfTO, VVfVV, DTfDT)</td>
      <td>(TOfTO, VVfVV, DTfDT)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(PPfPP, MDfMD, VVfVV)</td>
      <td>(NNfNN, INfIN, NNfNN)</td>
      <td>(JJfJJ, NNfNN, INfIN)</td>
      <td>(DTfDT, NNfNN, NNfNN)</td>
      <td>(DTfDT, NNfNN, NNfNN)</td>
      <td>(PPfPP, VVPfVVP, RBfRB)</td>
      <td>(PPfPP, MDfMD, VVfVV)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(DTfDT, NNfNN, NNfNN)</td>
      <td>(JJfJJ, NNfNN, INfIN)</td>
      <td>(PPfPP, VVPfVVP, RBfRB)</td>
      <td>(CDfCD, CDfCD, CDfCD)</td>
      <td>(PPfPP, VVPfVVP, RBfRB)</td>
      <td>(INfIN, DTfDT, NNSfNNS)</td>
      <td>(DTfDT, JJfJJ, NNSfNNS)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(PPfPP, VVPfVVP, PPfPP)</td>
      <td>(PPfPP, VVPfVVP, PPfPP)</td>
      <td>(INfIN, DTfDT, NNSfNNS)</td>
      <td>(NNfNN, NNfNN, INfIN)</td>
      <td>(INfIN, DTfDT, NNSfNNS)</td>
      <td>(JJfJJ, NNfNN, INfIN)</td>
      <td>(DTfDT, NNfNN, CCfCC)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(NNfNN, INfIN, PPfPP)</td>
      <td>(PPfPP, VVPfVVP, RBfRB)</td>
      <td>(DTfDT, NNfNN, CCfCC)</td>
      <td>(JJfJJ, NNfNN, INfIN)</td>
      <td>(JJfJJ, NNfNN, INfIN)</td>
      <td>(DTfDT, NNfNN, CCfCC)</td>
      <td>(NNfNN, INfIN, NNfNN)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(INfIN, INfIN, INfIN)</td>
      <td>(DTfDT, NNfNN, NNfNN)</td>
      <td>(DTfDT, NNfNN, NNfNN)</td>
      <td>(NNfNN, REfRE, REfRE)</td>
      <td>(NNfNN, INfIN, NNfNN)</td>
      <td>(NNfNN, INfIN, NNfNN)</td>
      <td>(NNSfNNS, INfIN, DTfDT)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(NNfNN, INfIN, NNfNN)</td>
      <td>(PPfPP, VBSfVBS, RBfRB)</td>
      <td>(PPfPP, VVPfVVP, PPfPP)</td>
      <td>(NNfNN, NNfNN, NNfNN)</td>
      <td>(VVfVV, INfIN, DTfDT)</td>
      <td>(NNSfNNS, INfIN, DTfDT)</td>
      <td>(DTfDT, NNfNN, NNfNN)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(INfIN, INfIN, DTfDT)</td>
      <td>(INfIN, DTfDT, NNSfNNS)</td>
      <td>(RBfRB, INfIN, DTfDT)</td>
      <td>(NNfNN, INfIN, NNfNN)</td>
      <td>(PPfPP, VVPfVVP, INfIN)</td>
      <td>(PPfPP, VVPfVVP, PPfPP)</td>
      <td>(DTfDT, NNSfNNS, INfIN)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(PPfPP, VVPfVVP, RBfRB)</td>
      <td>(RBfRB, INfIN, DTfDT)</td>
      <td>(PPfPP, VVPfVVP, INfIN)</td>
      <td>(PPfPP, VVPfVVP, RBfRB)</td>
      <td>(DTfDT, NNfNN, CCfCC)</td>
      <td>(INfIN, PPfPP, VVPfVVP)</td>
      <td>(DTfDT, NNfNN, PPfPP)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(INfIN, PPfPP, VVPfVVP)</td>
      <td>(PPfPP, VBPfVBP, VVGfVVG)</td>
      <td>(NNfNN, INfIN, NNfNN)</td>
      <td>(DTfDT, NNfNN, PPfPP)</td>
      <td>(VVfVV, DTfDT, NNfNN)</td>
      <td>(PPfPP, VVPfVVP, INfIN)</td>
      <td>(INfIN, DTfDT, NNSfNNS)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(INfIN, DTfDT, NNSfNNS)</td>
      <td>(DTfDT, NNfNN, CCfCC)</td>
      <td>(NNSfNNS, INfIN, DTfDT)</td>
      <td>(TOfTO, VVfVV, DTfDT)</td>
      <td>(NNfNN, NNfNN, INfIN)</td>
      <td>(VVfVV, DTfDT, NNfNN)</td>
      <td>(INfIN, JJfJJ, NNfNN)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(PPfPP, MDfMD, RBfRB)</td>
      <td>(TOfTO, VVfVV, INfIN)</td>
      <td>(INfIN, INfIN, DTfDT)</td>
      <td>(NNfNN, INfIN, NPfNP)</td>
      <td>(MDfMD, VVfVV, INfIN)</td>
      <td>(RBfRB, RBfRB, RBfRB)</td>
      <td>(VVfVV, DTfDT, NNfNN)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(DTfDT, NNfNN, CCfCC)</td>
      <td>(INfIN, PPfPP, VVPfVVP)</td>
      <td>(INfIN, PPfPP, VVPfVVP)</td>
      <td>(NNfNN, INfIN, PPfPP)</td>
      <td>(VVPfVVP, RBfRB, VVfVV)</td>
      <td>(DTfDT, NNfNN, NNfNN)</td>
      <td>(NNfNN, INfIN, JJfJJ)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(NNfNN, INfIN, INfIN)</td>
      <td>(DTfDT, NNfNN, PPfPP)</td>
      <td>(PPfPP, VBSfVBS, RBfRB)</td>
      <td>(REfRE, REfRE, PPfPP)</td>
      <td>(NNSfNNS, INfIN, DTfDT)</td>
      <td>(DTfDT, NNfNN, PPfPP)</td>
      <td>(JJfJJ, NNSfNNS, INfIN)</td>
    </tr>
  </tbody>
</table>
</div>



### Most common part-of-speech trigrams in Engish compare to other L1s


```python
#For each trigram in the common English trigram, calculate its proportion for each of the seven languages
eng_top_trigrams = [t[0] for t in eng_freq.most_common(15)]

eng_trigrams_df = pd.DataFrame(index=eng_top_trigrams, columns = ["L1=English", "L1=Danish", "L1=Finnish", "L1=Korean", "L1=Polish", "L1=Portuguese", "L1=Turkish"])

for trigram in eng_top_trigrams:
    eng_trigrams_df["L1=English"][trigram]= eng_freq[trigram]/len(eng_tag_trigrams)
    eng_trigrams_df["L1=Danish"][trigram]= dan_freq[trigram]/len(dan_tag_trigrams)
    eng_trigrams_df["L1=Finnish"][trigram]= fin_freq[trigram]/len(fin_tag_trigrams)
    eng_trigrams_df["L1=Korean"][trigram]= kor_freq[trigram]/len(kor_tag_trigrams)
    eng_trigrams_df["L1=Polish"][trigram]= pol_freq[trigram]/len(pol_tag_trigrams)
    eng_trigrams_df["L1=Portuguese"][trigram]= por_freq[trigram]/len(por_tag_trigrams)
    eng_trigrams_df["L1=Turkish"][trigram]= tur_freq[trigram]/len(tur_tag_trigrams)
    
eng_trigrams_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L1=English</th>
      <th>L1=Danish</th>
      <th>L1=Finnish</th>
      <th>L1=Korean</th>
      <th>L1=Polish</th>
      <th>L1=Portuguese</th>
      <th>L1=Turkish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(INfIN, DTfDT, NNfNN)</th>
      <td>0.0149823</td>
      <td>0.0156791</td>
      <td>0.0155236</td>
      <td>0.0107851</td>
      <td>0.0193401</td>
      <td>0.0175261</td>
      <td>0.0199672</td>
    </tr>
    <tr>
      <th>(DTfDT, NNfNN, INfIN)</th>
      <td>0.0120859</td>
      <td>0.0138714</td>
      <td>0.0124894</td>
      <td>0.008134</td>
      <td>0.0115941</td>
      <td>0.0148972</td>
      <td>0.0153654</td>
    </tr>
    <tr>
      <th>(DTfDT, JJfJJ, NNfNN)</th>
      <td>0.00937695</td>
      <td>0.0122731</td>
      <td>0.011184</td>
      <td>0.00897753</td>
      <td>0.00918593</td>
      <td>0.0100438</td>
      <td>0.0135715</td>
    </tr>
    <tr>
      <th>(NNfNN, INfIN, DTfDT)</th>
      <td>0.00762659</td>
      <td>0.00920957</td>
      <td>0.0101609</td>
      <td>0.0066277</td>
      <td>0.0103528</td>
      <td>0.0101786</td>
      <td>0.0116996</td>
    </tr>
    <tr>
      <th>(REfRE, REfRE, REfRE)</th>
      <td>0.00760575</td>
      <td>0.0116452</td>
      <td>0.0120308</td>
      <td>0.0204856</td>
      <td>0.012612</td>
      <td>0.003303</td>
      <td>0.00194993</td>
    </tr>
    <tr>
      <th>(PPfPP, MDfMD, VVfVV)</th>
      <td>0.00562617</td>
      <td>0.00477604</td>
      <td>0.00490404</td>
      <td>0.00560342</td>
      <td>0.00769632</td>
      <td>0.00795416</td>
      <td>0.0053038</td>
    </tr>
    <tr>
      <th>(INfIN, DTfDT, JJfJJ)</th>
      <td>0.00479267</td>
      <td>0.0067169</td>
      <td>0.00687976</td>
      <td>0.00289209</td>
      <td>0.00640532</td>
      <td>0.00640377</td>
      <td>0.00904766</td>
    </tr>
    <tr>
      <th>(NNfNN, INfIN, NNfNN)</th>
      <td>0.00425089</td>
      <td>0.00355824</td>
      <td>0.0033164</td>
      <td>0.00373561</td>
      <td>0.00397229</td>
      <td>0.00471857</td>
      <td>0.00452383</td>
    </tr>
    <tr>
      <th>(JJfJJ, NNfNN, INfIN)</th>
      <td>0.00420921</td>
      <td>0.0063744</td>
      <td>0.00483347</td>
      <td>0.00451889</td>
      <td>0.00402195</td>
      <td>0.00532524</td>
      <td>0.00748772</td>
    </tr>
    <tr>
      <th>(PPfPP, VVPfVVP, PPfPP)</th>
      <td>0.00406335</td>
      <td>0.00422423</td>
      <td>0.00381033</td>
      <td>0.00265108</td>
      <td>0.00206063</td>
      <td>0.00444894</td>
      <td>0.00249591</td>
    </tr>
    <tr>
      <th>(PPfPP, VVPfVVP, RBfRB)</th>
      <td>0.00391748</td>
      <td>0.00348213</td>
      <td>0.00472763</td>
      <td>0.00355486</td>
      <td>0.00427022</td>
      <td>0.00572969</td>
      <td>0.00319788</td>
    </tr>
    <tr>
      <th>(DTfDT, NNfNN, NNfNN)</th>
      <td>0.00387581</td>
      <td>0.00475701</td>
      <td>0.00402202</td>
      <td>0.00536241</td>
      <td>0.00486606</td>
      <td>0.00357263</td>
      <td>0.00413384</td>
    </tr>
    <tr>
      <th>(PPfPP, VBSfVBS, RBfRB)</th>
      <td>0.00387581</td>
      <td>0.00270198</td>
      <td>0.00310471</td>
      <td>0.00216907</td>
      <td>0.00258199</td>
      <td>0.0024941</td>
      <td>0.00233991</td>
    </tr>
    <tr>
      <th>(INfIN, DTfDT, NNSfNNS)</th>
      <td>0.00366743</td>
      <td>0.00344408</td>
      <td>0.00426898</td>
      <td>0.00114479</td>
      <td>0.00404677</td>
      <td>0.00572969</td>
      <td>0.00366586</td>
    </tr>
    <tr>
      <th>(RBfRB, INfIN, DTfDT)</th>
      <td>0.00329235</td>
      <td>0.00274004</td>
      <td>0.00377505</td>
      <td>0.0015063</td>
      <td>0.00273095</td>
      <td>0.00296596</td>
      <td>0.00296389</td>
    </tr>
  </tbody>
</table>
</div>




```python
eng_trigram_df = {}
eng_freq.most_common(20)
```




    [(('INfIN', 'DTfDT', 'NNfNN'), 719),
     (('DTfDT', 'NNfNN', 'INfIN'), 580),
     (('DTfDT', 'JJfJJ', 'NNfNN'), 450),
     (('NNfNN', 'INfIN', 'DTfDT'), 366),
     (('REfRE', 'REfRE', 'REfRE'), 365),
     (('PPfPP', 'MDfMD', 'VVfVV'), 270),
     (('INfIN', 'DTfDT', 'JJfJJ'), 230),
     (('NNfNN', 'INfIN', 'NNfNN'), 204),
     (('JJfJJ', 'NNfNN', 'INfIN'), 202),
     (('PPfPP', 'VVPfVVP', 'PPfPP'), 195),
     (('PPfPP', 'VVPfVVP', 'RBfRB'), 188),
     (('DTfDT', 'NNfNN', 'NNfNN'), 186),
     (('PPfPP', 'VBSfVBS', 'RBfRB'), 186),
     (('INfIN', 'DTfDT', 'NNSfNNS'), 176),
     (('RBfRB', 'INfIN', 'DTfDT'), 158),
     (('PPfPP', 'VBPfVBP', 'VVGfVVG'), 157),
     (('DTfDT', 'NNfNN', 'CCfCC'), 154),
     (('TOfTO', 'VVfVV', 'INfIN'), 151),
     (('INfIN', 'PPfPP', 'VVPfVVP'), 148),
     (('DTfDT', 'NNfNN', 'PPfPP'), 144)]



### Get native speaker trigram outliers (in terms of frequency vs. other L1s)


```python
#For each trigram in the common English trigram, calculate its average frequency among the non-native
#speakers. If the trigram's frequency in English is 1.5 times greater than this average, add it to the more_common
#dictionary. If the trigram's frequency in English is 15 times less than this average, add it to the
#less_common dictionaries.
more_common = {}
less_common = {}

for trigram in eng_freq:
    #trigram = trigram[0]
    
    avg = 0
    avg = avg + (kor_freq[trigram] / len(kor_tag_trigrams))
    avg = avg + (fin_freq[trigram] / len(fin_tag_trigrams))
    avg = avg + (tur_freq[trigram] / len(tur_tag_trigrams))
    avg = avg + (dan_freq[trigram] / len(dan_tag_trigrams))
    avg = avg + (por_freq[trigram] / len(por_tag_trigrams))
    avg = avg + (pol_freq[trigram] / len(pol_tag_trigrams))
    
    avg /= 6
    
    eng_percent = (eng_freq[trigram] / len(eng_tag_trigrams))
    if eng_percent > (avg * 2) and eng_freq[trigram] > 5:
        more_common[trigram] = eng_freq[trigram]
        
        
    if eng_percent < (avg * 0.5) and eng_freq[trigram] > 5:
        less_common[trigram] = eng_freq[trigram]
```

#### Native trigram outliers -- more frequent that other L1s.


```python
#Show the trigrams that are particularly more common among the native English than the non-native speakers
for trigram in sorted(more_common, key=more_common.get, reverse=True)[:10]:
    print(trigram)
    
    print("\tEnglish: " + str(eng_freq[trigram] / len(eng_tag_trigrams)))
    print("\tFinnish: " + str(fin_freq[trigram] / len(fin_tag_trigrams)))
    print("\tKorean: " + str(kor_freq[trigram] / len(kor_tag_trigrams)))
    print("\tTurkish: " + str(tur_freq[trigram] / len(tur_tag_trigrams)))
    print("\tDanish: " + str(dan_freq[trigram] / len(dan_tag_trigrams)))
    print("\tPortuguese: " + str(por_freq[trigram] / len(por_tag_trigrams)))
    print("\tPolish: " + str(pol_freq[trigram] / len(pol_tag_trigrams)))
```

    ('PPfPP', 'VHPfVHP', 'VVNfVVN')
    	English: 0.002208793498645551
    	Finnish: 0.00091730172170477
    	Korean: 0.0006627703801891908
    	Turkish: 0.0006239762889010217
    	Danish: 0.0012748791718993797
    	Portuguese: 0.0006066734074823053
    	Polish: 0.001390302639092331
    ('PPfPP', 'VBDfVBD', 'VVGfVVG')
    	English: 0.0013752865180245884
    	Finnish: 0.0003175275190516511
    	Korean: 6.0251852744471894e-05
    	Turkish: 0.00015599407222525544
    	Danish: 0.0004186170415191993
    	Portuguese: 0.0013481631277384564
    	Polish: 0.0004965366568186896
    ('VBPfVBP', 'VVGfVVG', 'INfIN')
    	English: 0.0013336111689935403
    	Finnish: 0.0009525825571549535
    	Korean: 6.0251852744471894e-05
    	Turkish: 0.0007019733250136495
    	Danish: 0.0006088975149370171
    	Portuguese: 0.0009437141894169194
    	Polish: 0.0007199781523870999
    ('DTfDT', 'VBSfVBS', 'RBfRB')
    	English: 0.0010418837257762034
    	Finnish: 0.0007056167090036692
    	Korean: 0.00030125926372235944
    	Turkish: 7.799703611262772e-05
    	Danish: 0.000799177988354835
    	Portuguese: 0.0
    	Polish: 0.0003972293254549517
    ('PREfPRE', 'PPfPP', 'VVPfVVP')
    	English: 0.0007709939570743905
    	Finnish: 0.0006703358735534857
    	Korean: 0.0
    	Turkish: 0.00023399110833788317
    	Danish: 0.00032347680481029036
    	Portuguese: 0.0004044489383215369
    	Polish: 0.0002730951612502793
    ('VBSfVBS', 'RBfRB', 'DTfDT')
    	English: 0.0007501562825588665
    	Finnish: 0.0006703358735534857
    	Korean: 0.00030125926372235944
    	Turkish: 0.00015599407222525544
    	Danish: 0.0005137572782281082
    	Portuguese: 0.00020222446916076846
    	Polish: 0.00019861466272747585
    ('VBDfVBD', 'VVGfVVG', 'INfIN')
    	English: 0.0006459679099812461
    	Finnish: 0.00010584250635055038
    	Korean: 6.0251852744471894e-05
    	Turkish: 0.00015599407222525544
    	Danish: 0.00011416828405069072
    	Portuguese: 0.0004044489383215369
    	Polish: 0.00017378782988654139
    ('INfIN', 'WPfWP', 'PPfPP')
    	English: 0.000604292560950198
    	Finnish: 0.0001764041772509173
    	Korean: 0.00012050370548894379
    	Turkish: 0.00015599407222525544
    	Danish: 0.0006659816569623625
    	Portuguese: 0.0004718570947084597
    	Polish: 0.00019861466272747585
    ('INfIN', 'PPfPP', 'REfRE')
    	English: 0.000604292560950198
    	Finnish: 0.00021168501270110075
    	Korean: 0.00036151111646683137
    	Turkish: 0.00015599407222525544
    	Danish: 0.00030444875746850857
    	Portuguese: 0.0002696326255476913
    	Polish: 0.00022344149556841034
    ('NNSfNNS', 'PREfPRE', 'VVPfVVP')
    	English: 0.000604292560950198
    	Finnish: 0.0002469658481512842
    	Korean: 0.00018075555823341568
    	Turkish: 7.799703611262772e-05
    	Danish: 0.00020930852075959965
    	Portuguese: 0.0004044489383215369
    	Polish: 0.0002482683284093448


#### Native trigram outliers -- less frequent that other L1s. (Maybe try this again, but excluding pause and hesitation tags)


```python
#Show the trigrams that are particularly less frequent among the native English than the non-native speakers
for trigram in sorted(less_common, key=less_common.get, reverse=True)[:10]:
    print(trigram)
    print("\tEnglish: " + str(eng_freq[trigram] / len(eng_tag_trigrams)))
    print("\tFinnish: " + str(fin_freq[trigram] / len(fin_tag_trigrams)))
    print("\tKorean: " + str(kor_freq[trigram] / len(kor_tag_trigrams)))
    print("\tTurkish: " + str(tur_freq[trigram] / len(tur_tag_trigrams)))
    print("\tDanish: " + str(dan_freq[trigram] / len(dan_tag_trigrams)))
    print("\tPortuguese: " + str(por_freq[trigram] / len(por_tag_trigrams)))
    print("\tPolish: " + str(pol_freq[trigram] / len(pol_tag_trigrams)))
```

    ('PPfPP', 'VHPfVHP', 'DTfDT')
    	English: 0.0006876432590122942
    	Finnish: 0.0010231442280553204
    	Korean: 0.0014460444658673255
    	Turkish: 0.001403946650027299
    	Danish: 0.0018266925448110515
    	Portuguese: 0.0019548365352207615
    	Polish: 0.0008192854837508379
    ('PPfPP', 'PPfPP', 'PPfPP')
    	English: 0.0004584288393415295
    	Finnish: 0.000458650860852385
    	Korean: 0.0007230222329336627
    	Turkish: 0.00023399110833788317
    	Danish: 0.001788636450127488
    	Portuguese: 0.0018874283788338389
    	Polish: 0.0005710171553414931
    ('NPfNP', 'CCfCC', 'NPfNP')
    	English: 0.0004375911648260054
    	Finnish: 0.0005644933672029354
    	Korean: 0.0025908296680122915
    	Turkish: 0.0012479525778020435
    	Danish: 0.0005137572782281082
    	Portuguese: 0.0010785305021907652
    	Polish: 0.0005958439881824276
    ('CCfCC', 'DTfDT', 'JJfJJ')
    	English: 0.0003959158157949573
    	Finnish: 0.0011289867344058708
    	Korean: 0.00030125926372235944
    	Turkish: 0.001949925902815693
    	Danish: 0.0007611218936712714
    	Portuguese: 0.0009437141894169194
    	Polish: 0.0004468829911368207
    ('DTfDT', 'NPfNP', 'NNfNN')
    	English: 0.00037507814127943324
    	Finnish: 0.0006703358735534857
    	Korean: 0.0008435259384226065
    	Turkish: 0.000545979252788394
    	Danish: 0.001617384024051452
    	Portuguese: 0.0004718570947084597
    	Polish: 0.0007448049852280344
    ('RBfRB', 'PPfPP', 'VHPfVHP')
    	English: 0.00037507814127943324
    	Finnish: 0.0005644933672029354
    	Korean: 0.0007832740856781347
    	Turkish: 0.0010139614694641603
    	Danish: 0.0008182060356966168
    	Portuguese: 0.0005392652510953826
    	Polish: 0.0008192854837508379
    ('NPfNP', 'CCfCC', 'PPfPP')
    	English: 0.0003542404667639091
    	Finnish: 0.0003880891899520181
    	Korean: 0.0009037777911670784
    	Turkish: 0.0006239762889010217
    	Danish: 0.000799177988354835
    	Portuguese: 0.0008088978766430738
    	Polish: 0.0007696318180689689
    ('EXfEX', 'VBZfVBZ', 'DTfDT')
    	English: 0.0003542404667639091
    	Finnish: 0.0005292125317527519
    	Korean: 0.00048201482195577515
    	Turkish: 0.0012479525778020435
    	Danish: 0.00053278532556989
    	Portuguese: 0.0006740815638692282
    	Polish: 0.0011420343106829862
    ('CDfCD', 'CDfCD', 'CDfCD')
    	English: 0.00033340279224838506
    	Finnish: 0.00010584250635055038
    	Korean: 0.00475989636681328
    	Turkish: 0.0012479525778020435
    	Danish: 0.000266392662784945
    	Portuguese: 0.0008088978766430738
    	Polish: 0.0010179001464783138
    ('FWfFW', 'FWfFW', 'FWfFW')
    	English: 0.00033340279224838506
    	Finnish: 3.528083545018346e-05
    	Korean: 0.006266192685425077
    	Turkish: 0.0
    	Danish: 0.0008182060356966168
    	Portuguese: 6.740815638692282e-05
    	Polish: 0.002333722287047841


## Comparing discourse markers


```python
def get_discourse_markers(speech):
    dm_list = []
    markers = [[t for t in u if t[1] == "DMfDM"] for u in speech]
    for m in markers:
        dm_list.extend(m)
        
    #return dm_list
    return dm_list
```


```python
#Get the discourse markers for each L1 group
eng_dm = get_discourse_markers(eng_speech)
kor_dm = get_discourse_markers(kor_speech)
fin_dm = get_discourse_markers(fin_speech)
tur_dm = get_discourse_markers(tur_speech)
dan_dm = get_discourse_markers(dan_speech)
por_dm = get_discourse_markers(por_speech)
pol_dm = get_discourse_markers(pol_speech)
```


```python
eng_dm[:20]
```




    [('so', 'DMfDM'),
     ('well', 'DMfDM'),
     ('so', 'DMfDM'),
     ('like', 'DMfDM'),
     ('so', 'DMfDM'),
     ('well', 'DMfDM'),
     ('so', 'DMfDM'),
     ('so', 'DMfDM'),
     ('so', 'DMfDM'),
     ('so', 'DMfDM'),
     ('so', 'DMfDM'),
     ('\n\nso\n\n', 'DMfDM'),
     ('\n\nso\n\n', 'DMfDM'),
     ('so', 'DMfDM'),
     ('\n\nlike\n\n', 'DMfDM'),
     ('like', 'DMfDM'),
     ('like', 'DMfDM'),
     ('so', 'DMfDM'),
     ('like', 'DMfDM'),
     ('like', 'DMfDM')]



### Percent of discourse markers across L1s


```python
#Counts the number of discourse markers, and get the proportion by dividing by it by the total
#number of words
def get_dm_percent(speech, dm_list):
    total = 0
    for u in speech:
        total += len(u)
    
    return len(dm_list)/total
```


```python
#Create a data frame with the proportion/frequencies of discourse markers across each L1 group
dm_df = pd.DataFrame(index = ['determiner proportions'], columns = ['L1=English', 'L1=Korean', 'L1=Finnish', 'L1=Turkish', 'L1=Danish', 'L1=Portuguese', 'L1=Polish'])

dm_df['L1=English'] = get_dm_percent(eng_speech, eng_dm)
dm_df['L1=Korean'] = get_dm_percent(kor_speech, kor_dm)
dm_df['L1=Finnish'] = get_dm_percent(fin_speech, fin_dm)
dm_df['L1=Turkish'] = get_dm_percent(tur_speech, tur_dm)
dm_df['L1=Danish'] = get_dm_percent(dan_speech, dan_dm)
dm_df['L1=Portuguese'] = get_dm_percent(por_speech, por_dm)
dm_df['L1=Polish'] = get_dm_percent(pol_speech, pol_dm)

dm_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L1=English</th>
      <th>L1=Korean</th>
      <th>L1=Finnish</th>
      <th>L1=Turkish</th>
      <th>L1=Danish</th>
      <th>L1=Portuguese</th>
      <th>L1=Polish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>determiner proportions</th>
      <td>0.014251</td>
      <td>0.013229</td>
      <td>0.012288</td>
      <td>0.010293</td>
      <td>0.01287</td>
      <td>0.008287</td>
      <td>0.017388</td>
    </tr>
  </tbody>
</table>
</div>




```python
eng_dm_words = [dm[0].replace('\n', '') for dm in eng_dm]
kor_dm_words = [dm[0].replace('\n', '') for dm in kor_dm]
fin_dm_words = [dm[0].replace('\n', '') for dm in fin_dm]
tur_dm_words = [dm[0].replace('\n', '') for dm in tur_dm]
dan_dm_words = [dm[0].replace('\n', '') for dm in dan_dm]
por_dm_words = [dm[0].replace('\n', '') for dm in por_dm]
pol_dm_words = [dm[0].replace('\n', '') for dm in pol_dm]
```


```python
eng_dm_freqs = nltk.FreqDist(eng_dm_words)
eng_dm_freqs
```




    FreqDist({'like': 202,
              'look': 1,
              'right': 50,
              'so': 419,
              'well': 116,
              'whatever': 13})




```python
kor_dm_freqs = nltk.FreqDist(kor_dm_words)
kor_dm_freqs
```




    FreqDist({'like': 55, 'right': 33, 'so': 174, 'well': 27, 'whatever': 4})




```python
fin_dm_freqs = nltk.FreqDist(fin_dm_words)
fin_dm_freqs
```




    FreqDist({'like': 111, 'right': 10, 'so': 226, 'well': 72, 'whatever': 8})




```python
tur_dm_freqs = nltk.FreqDist(tur_dm_words)
tur_dm_freqs
```




    FreqDist({'like': 16, 'right': 15, 'so': 119, 'well': 13, 'whatever': 1})




```python
dan_dm_freqs = nltk.FreqDist(dan_dm_words)
dan_dm_freqs
```




    FreqDist({'like': 125, 'right': 30, 'so': 499, 'well': 137, 'whatever': 16})




```python
pol_dm_freqs = nltk.FreqDist(pol_dm_words)
pol_dm_freqs
```




    FreqDist({'like': 265,
              'look': 1,
              'right': 40,
              'so': 450,
              'well': 66,
              'whatever': 28})




```python
por_dm_freqs = nltk.FreqDist(por_dm_words)
por_dm_freqs
```




    FreqDist({'like': 18, 'right': 3, 'so': 114, 'well': 15, 'whatever': 3})



### Getting distribution of specific discourse markers across specific L1s


```python
#Get the frequencies of each discourse marker (like, look, right, so, well, whatever), for each L1
discourse_markers = eng_dm_freqs.most_common()
discourse_markers = [b[0] for b in discourse_markers]

dm_words_df = pd.DataFrame(index = discourse_markers, columns = ['L1=English', 'L1=Korean', 'L1=Finnish', 'L1=Turkish', 'L1=Danish', 'L1=Portuguese', 'L1=Polish'])

for word in discourse_markers:
    dm_words_df['L1=English'][word] = eng_dm_freqs[word]/sum(eng_dm_freqs.values())
    dm_words_df['L1=Korean'][word] = kor_dm_freqs[word]/sum(kor_dm_freqs.values())
    dm_words_df['L1=Finnish'][word] = fin_dm_freqs[word]/sum(fin_dm_freqs.values())
    dm_words_df['L1=Turkish'][word] = tur_dm_freqs[word]/sum(tur_dm_freqs.values())
    dm_words_df['L1=Danish'][word] = dan_dm_freqs[word]/sum(dan_dm_freqs.values())
    dm_words_df['L1=Portuguese'][word] = por_dm_freqs[word]/sum(por_dm_freqs.values())
    dm_words_df['L1=Polish'][word] = pol_dm_freqs[word]/sum(pol_dm_freqs.values())

    
dm_words_df.plot(kind='bar', figsize=(15,10))
plt.tick_params(axis = 'both', reset=True, labelsize=15)
plt.legend(prop={'size':20})
bar_width = 0.4
L=plt.legend()
plt.xlabel("Determiner", fontsize= 20, labelpad = 20)
plt.ylabel("Frequency (percent)", fontsize= 20, labelpad = 20)
plt.show()    
    

dm_words_df
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12a92ab38>






    <matplotlib.legend.Legend at 0x12dca86a0>






    <matplotlib.text.Text at 0x12a934e48>






    <matplotlib.text.Text at 0x12a969518>




![png](output_46_4.png)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L1=English</th>
      <th>L1=Korean</th>
      <th>L1=Finnish</th>
      <th>L1=Turkish</th>
      <th>L1=Danish</th>
      <th>L1=Portuguese</th>
      <th>L1=Polish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>so</th>
      <td>0.523096</td>
      <td>0.593857</td>
      <td>0.529274</td>
      <td>0.72561</td>
      <td>0.61834</td>
      <td>0.745098</td>
      <td>0.529412</td>
    </tr>
    <tr>
      <th>like</th>
      <td>0.252185</td>
      <td>0.187713</td>
      <td>0.259953</td>
      <td>0.097561</td>
      <td>0.154895</td>
      <td>0.117647</td>
      <td>0.311765</td>
    </tr>
    <tr>
      <th>well</th>
      <td>0.144819</td>
      <td>0.0921502</td>
      <td>0.168618</td>
      <td>0.0792683</td>
      <td>0.169765</td>
      <td>0.0980392</td>
      <td>0.0776471</td>
    </tr>
    <tr>
      <th>right</th>
      <td>0.062422</td>
      <td>0.112628</td>
      <td>0.0234192</td>
      <td>0.0914634</td>
      <td>0.0371747</td>
      <td>0.0196078</td>
      <td>0.0470588</td>
    </tr>
    <tr>
      <th>whatever</th>
      <td>0.0162297</td>
      <td>0.0136519</td>
      <td>0.0187354</td>
      <td>0.00609756</td>
      <td>0.0198265</td>
      <td>0.0196078</td>
      <td>0.0329412</td>
    </tr>
    <tr>
      <th>look</th>
      <td>0.00124844</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00117647</td>
    </tr>
  </tbody>
</table>
</div>



## Comparing Article/Determiner Use

### Proportion of determiners


```python
#Get the total number of words for the L1 text
def num_words(li):
    count = 0
    for u in li:
        count += len(u)
    
    return count
```


```python
#Return the number of determiners in the list passed in
def determiner_words(li):
    words = []
    
    for utterance in li:
        for pair in utterance:
            if pair[1] == "DTfDT":
                words.append(pair[0].replace("\n", ""))
    
    return words
```


```python
#Create a data frame containing the percentage of determiners for each L1 group

det_df = pd.DataFrame(index = ['determiner proportions'], columns = ['L1=English', 'L1=Korean', 'L1=Finnish', 'L1=Turkish', 'L1=Danish', 'L1=Portuguese', 'L1=Polish'])

eng_dets = determiner_words(eng_speech)
kor_dets = determiner_words(kor_speech)
fin_dets = determiner_words(fin_speech)
tur_dets = determiner_words(tur_speech)
dan_dets = determiner_words(dan_speech)
por_dets = determiner_words(por_speech)
pol_dets = determiner_words(pol_speech)

det_df['L1=English'] = len(eng_dets)/len(eng_toks)
det_df['L1=Korean'] = len(kor_dets)/len(kor_toks)
det_df['L1=Finnish'] = len(fin_dets)/len(fin_toks)
det_df['L1=Turkish'] = len(tur_dets)/len(tur_toks)
det_df['L1=Danish'] = len(dan_dets)/len(dan_toks)
det_df['L1=Portuguese'] = len(por_dets)/len(por_toks)
det_df['L1=Polish'] = len(pol_dets)/len(pol_toks)

det_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L1=English</th>
      <th>L1=Korean</th>
      <th>L1=Finnish</th>
      <th>L1=Turkish</th>
      <th>L1=Danish</th>
      <th>L1=Portuguese</th>
      <th>L1=Polish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>determiner proportions</th>
      <td>0.092807</td>
      <td>0.085427</td>
      <td>0.107176</td>
      <td>0.106683</td>
      <td>0.106743</td>
      <td>0.107973</td>
      <td>0.094908</td>
    </tr>
  </tbody>
</table>
</div>



### Comparing specfic determiner words


```python
eng_det_words = nltk.FreqDist(eng_dets)
kor_det_words = nltk.FreqDist(kor_dets)
fin_det_words = nltk.FreqDist(fin_dets)
tur_det_words = nltk.FreqDist(tur_dets)
dan_det_words = nltk.FreqDist(dan_dets)
por_det_words = nltk.FreqDist(por_dets)
pol_det_words = nltk.FreqDist(pol_dets)
```


```python
#Create a dataframe showing the frequencies of each determiner for each L1
det_words = eng_det_words.most_common()
det_words = [b[0] for b in det_words]

det_words_df = pd.DataFrame(index = det_words, columns = ['L1=English', 'L1=Korean', 'L1=Finnish', 'L1=Turkish', 'L1=Danish', 'L1=Portuguese', 'L1=Polish'])

for word in det_words:
    det_words_df['L1=English'][word] = eng_det_words[word]/len(eng_toks)
    det_words_df['L1=Korean'][word] = kor_det_words[word]/len(kor_toks)
    det_words_df['L1=Finnish'][word] = fin_det_words[word]/len(fin_toks)
    det_words_df['L1=Turkish'][word] = tur_det_words[word]/len(tur_toks)
    det_words_df['L1=Danish'][word] = dan_det_words[word]/len(dan_toks)
    det_words_df['L1=Portuguese'][word] = por_det_words[word]/len(por_toks)
    det_words_df['L1=Polish'][word] = pol_det_words[word]/len(pol_toks)
det_words_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L1=English</th>
      <th>L1=Korean</th>
      <th>L1=Finnish</th>
      <th>L1=Turkish</th>
      <th>L1=Danish</th>
      <th>L1=Portuguese</th>
      <th>L1=Polish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>the</th>
      <td>0.0406318</td>
      <td>0.0398819</td>
      <td>0.0539406</td>
      <td>0.0626998</td>
      <td>0.0489002</td>
      <td>0.0549302</td>
      <td>0.0509173</td>
    </tr>
    <tr>
      <th>a</th>
      <td>0.0178155</td>
      <td>0.0127719</td>
      <td>0.0151697</td>
      <td>0.0140373</td>
      <td>0.0213677</td>
      <td>0.0163106</td>
      <td>0.011668</td>
    </tr>
    <tr>
      <th>that</th>
      <td>0.0147108</td>
      <td>0.0094584</td>
      <td>0.0138997</td>
      <td>0.0113858</td>
      <td>0.0148984</td>
      <td>0.00660511</td>
      <td>0.00638018</td>
    </tr>
    <tr>
      <th>this</th>
      <td>0.00683447</td>
      <td>0.00885596</td>
      <td>0.0102307</td>
      <td>0.00600484</td>
      <td>0.00808661</td>
      <td>0.0137494</td>
      <td>0.0140513</td>
    </tr>
    <tr>
      <th>some</th>
      <td>0.00254209</td>
      <td>0.00668715</td>
      <td>0.00225781</td>
      <td>0.00327536</td>
      <td>0.0026448</td>
      <td>0.00431354</td>
      <td>0.00245773</td>
    </tr>
    <tr>
      <th>an</th>
      <td>0.00245874</td>
      <td>0.000421712</td>
      <td>0.00201087</td>
      <td>0.00116977</td>
      <td>0.00323464</td>
      <td>0.00155018</td>
      <td>0.00109233</td>
    </tr>
    <tr>
      <th>all</th>
      <td>0.00195866</td>
      <td>0.00174709</td>
      <td>0.00201087</td>
      <td>0.00194962</td>
      <td>0.00213106</td>
      <td>0.00269596</td>
      <td>0.00181227</td>
    </tr>
    <tr>
      <th>these</th>
      <td>0.0018128</td>
      <td>0.000903669</td>
      <td>0.00268115</td>
      <td>0.0015597</td>
      <td>0.00121775</td>
      <td>0.00141538</td>
      <td>0.000744768</td>
    </tr>
    <tr>
      <th>any</th>
      <td>0.000958493</td>
      <td>0.00126514</td>
      <td>0.00105835</td>
      <td>0.00124776</td>
      <td>0.000932339</td>
      <td>0.00155018</td>
      <td>0.00131576</td>
    </tr>
    <tr>
      <th>those</th>
      <td>0.000812635</td>
      <td>0.000662691</td>
      <td>0.00116419</td>
      <td>0.00132574</td>
      <td>0.00078012</td>
      <td>0.00128058</td>
      <td>0.00139023</td>
    </tr>
    <tr>
      <th>another</th>
      <td>0.000770962</td>
      <td>0.000481957</td>
      <td>0.000458618</td>
      <td>0.000389924</td>
      <td>0.000799148</td>
      <td>0.000943587</td>
      <td>0.000546163</td>
    </tr>
    <tr>
      <th>no</th>
      <td>0.000666778</td>
      <td>0.000903669</td>
      <td>0.000917237</td>
      <td>0.000623879</td>
      <td>0.000951366</td>
      <td>0.000943587</td>
      <td>0.00084407</td>
    </tr>
    <tr>
      <th>each</th>
      <td>0.000375063</td>
      <td>0.000180734</td>
      <td>0.000493897</td>
      <td>0.000467909</td>
      <td>0.000247355</td>
      <td>0.000808789</td>
      <td>0.00022343</td>
    </tr>
    <tr>
      <th>such</th>
      <td>0.000229205</td>
      <td>0</td>
      <td>0.000141113</td>
      <td>0</td>
      <td>9.51366e-05</td>
      <td>0</td>
      <td>0.000248256</td>
    </tr>
    <tr>
      <th>both</th>
      <td>0.000166694</td>
      <td>0.000180734</td>
      <td>0.000458618</td>
      <td>0.000233955</td>
      <td>0.00028541</td>
      <td>0.000471793</td>
      <td>0.00022343</td>
    </tr>
    <tr>
      <th>every</th>
      <td>6.25104e-05</td>
      <td>0.00102416</td>
      <td>0.000282227</td>
      <td>0.000311939</td>
      <td>0.000133191</td>
      <td>0.000404394</td>
      <td>0.000968198</td>
    </tr>
  </tbody>
</table>
</div>


