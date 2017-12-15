

```python
import pickle
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
stopWords = set(stopwords.words('english'))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
f = open('VOICE_tokenized.p', 'rb')
VOICE_toks = pickle.load(f)
f.close()

f = open('BNC_tokenized.p', 'rb')
BNC_toks = pickle.load(f)
f.close()

f = open('VOICE_participant_info.p', 'rb')
participants = pickle.load(f)
f.close()
```

# Comparing Native and Non-native Speakers (VOICE and BNC)


```python
#Takes a conversation dictionary and a list of languages. Returns a dictionary of 
#conversations (subcorpus) containing only utterances from participants whose L1 is 
#included in the list of languages.

def speech_by_L1(conversation_dict, lang_list):
    new_dict = {}
    for conversation in conversation_dict.keys():
        utterances = {}
        
        for pair in conversation_dict[conversation]:    
            speaker = pair[1]
            #print(conversation_dict[conversation][pair])

            
            #print(participants[speaker]['L1'][0])
            if(participants[speaker]['L1'][0] in lang_list):
                utterances[pair] = conversation_dict[conversation][pair]
            

        new_dict[conversation] = utterances    

    return new_dict
```


```python
#Takes a dictionary of conversations from a corpus, and returns a list of bigrams.
def get_bigrams(dictionary):
    bigrams = []
    for file in dictionary:
        for key in dictionary[file]:
            pairs = list(nltk.bigrams(dictionary[file][key]))
            bigrams.extend(pairs)
    return bigrams
```


```python
#Takes a list of bigrams, returns dictionary whose keys are bigrams containing 
#duplicate words (e.g ('i', 'i',), ('the', 'the') and whose values are the 
#frequencies of each bigram

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


```python
#Creating subcorpora from the VOICE corpus: one for participants with Germanic L1s, one for
#participants with Romance L1s, and one for participants with Slavic L1s
lang_classifications = {'germanic': ['ger', 'dut', 'swe', 'dan', 'nor', 'ice'],
                        'romance': ['por', 'spa', 'ita', 'fre', 'cat', 'rum'],
                        'slavic': ['rus', 'ukr', 'pol', 'bul', 'mac', 'cze', 'bos', 'slo', 'slv']}

germanic_toks = speech_by_L1(VOICE_toks, lang_classifications['germanic'])
romance_toks = speech_by_L1(VOICE_toks, lang_classifications['romance'])
slavic_toks = speech_by_L1(VOICE_toks, lang_classifications['slavic'])
```


```python
#Get lists of bigrams for VOICE and BNC, in addition to the three sub-corpora
VOICE_bigrams = get_bigrams(VOICE_toks)
BNC_bigrams = get_bigrams(BNC_toks)
germanic_bigrams = get_bigrams(germanic_toks)
romance_bigrams = get_bigrams(romance_toks)
slavic_bigrams = get_bigrams(slavic_toks)
```

## Bigrams

### Hesitations and Repeated Words


```python
BNC_repeated_words = [b for b in BNC_bigrams if b[0] == b[1]]
VOICE_repeated_words = [b for b in VOICE_bigrams if b[0] == b[1]]
germanic_repeated_words = [b for b in germanic_bigrams if b[0] == b[1]]
romance_repeated_words = [b for b in romance_bigrams if b[0] == b[1]]
slavic_repeated_words = [b for b in slavic_bigrams if b[0] == b[1]]
```


```python
repeated_df = pd.DataFrame(index = ['repeated words'], columns = ['L1=English(BNC)', 'All(VOICE)', 'L1=Germanic(VOICE)', 'L1=Romance(VOICE)', 'L1=Slavic(VOICE)'])
repeated_df['L1=English(BNC)']['repeated words'] = len(BNC_repeated_words)/len(BNC_bigrams)
repeated_df['All(VOICE)']['repeated words'] = len(VOICE_repeated_words)/len(VOICE_bigrams)
repeated_df['L1=Germanic(VOICE)']['repeated words'] = len(germanic_repeated_words)/len(germanic_bigrams)
repeated_df['L1=Romance(VOICE)']['repeated words'] = len(romance_repeated_words)/len(romance_bigrams)
repeated_df['L1=Slavic(VOICE)']['repeated words'] = len(slavic_repeated_words)/len(slavic_bigrams)
```


```python
fig = repeated_df.plot(kind='bar', figsize=(10,5))
plt.tick_params(axis = 'both', reset=True, labelsize=15)
plt.legend(prop={'size':20})
bar_width = 0.4
L=plt.legend()
L.get_texts()[1].set_text('All non-native L1s(VOICE)')
plt.ylabel("Frequency (percent)", fontsize= 20, labelpad = 20)
plt.show()
```




    <matplotlib.legend.Legend at 0x1805668d0>






    <matplotlib.text.Text at 0x1805305c0>




![png](output_12_2.png)


### Contractions


```python
BNC_contractions = [b for b in BNC_bigrams if "'" in b[1]]
VOICE_contractions = [b for b in VOICE_bigrams if "'" in b[1]]
germanic_contractions = [b for b in germanic_bigrams if "'" in b[1]]
romance_contractions = [b for b in romance_bigrams if "'" in b[1]]
slavic_contractions = [b for b in slavic_bigrams if "'" in b[1]]
```


```python
contraction_proportion_df = pd.DataFrame(index = ['contraction proportions'], columns = ['L1=English(BNC)', 'All(VOICE)', 'L1=Germanic(VOICE)', 'L1=Romance(VOICE)', 'L1=Slavic(VOICE)'])

contraction_proportion_df['L1=English(BNC)']['contraction proportions'] = len(BNC_contractions)/len(BNC_bigrams)
contraction_proportion_df['All(VOICE)']['contraction proportions'] = len(VOICE_contractions)/len(VOICE_bigrams)
contraction_proportion_df['L1=Germanic(VOICE)']['contraction proportions'] = len(germanic_contractions)/len(germanic_bigrams)
contraction_proportion_df['L1=Romance(VOICE)']['contraction proportions'] = len(romance_contractions)/len(romance_bigrams)
contraction_proportion_df['L1=Slavic(VOICE)']['contraction proportions'] = len(slavic_contractions)/len(slavic_bigrams)
```


```python
contraction_proportion_df.plot(kind='bar', figsize=(10,5))
plt.tick_params(axis = 'both', reset=True, labelsize=15)
plt.legend(prop={'size':20})
bar_width = 0.4
L=plt.legend()
L.get_texts()[1].set_text('All non-native L1s(VOICE)')
plt.ylabel("Frequency (percent)", fontsize= 20, labelpad = 20)
plt.show()

contraction_proportion_df
```




    <matplotlib.axes._subplots.AxesSubplot at 0x188661748>






    <matplotlib.legend.Legend at 0x188cd8400>






    <matplotlib.text.Text at 0x1887d9668>




![png](output_16_3.png)





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
      <th>L1=English(BNC)</th>
      <th>All(VOICE)</th>
      <th>L1=Germanic(VOICE)</th>
      <th>L1=Romance(VOICE)</th>
      <th>L1=Slavic(VOICE)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>contraction proportions</th>
      <td>0.0536744</td>
      <td>0.0341021</td>
      <td>0.0350262</td>
      <td>0.0340637</td>
      <td>0.0307484</td>
    </tr>
  </tbody>
</table>
</div>




```python
BNC_contractions_frequencies = nltk.FreqDist(BNC_contractions)
germanic_contractions_frequencies = nltk.FreqDist(germanic_contractions)
romance_contractions_frequencies = nltk.FreqDist(romance_contractions)
slavic_contractions_frequencies = nltk.FreqDist(slavic_contractions)
```


```python
BNC_top_15 = BNC_contractions_frequencies.most_common(15)
BNC_top_15 = [b[0] for b in BNC_top_15]
bigrams_df = pd.DataFrame(index = BNC_top_15, columns = ['English(BNC)', 'Germanic(VOICE)', 'Romance(VOICE)', 'Slavic(VOICE)'])

for bigram in BNC_top_15:
    bigrams_df['English(BNC)'][bigram] = BNC_contractions_frequencies[bigram]/len(BNC_bigrams)
    bigrams_df['Germanic(VOICE)'][bigram] = germanic_contractions_frequencies[bigram]/len(germanic_bigrams)
    bigrams_df['Romance(VOICE)'][bigram] = romance_contractions_frequencies[bigram]/len(romance_bigrams)
    bigrams_df['Slavic(VOICE)'][bigram] = slavic_contractions_frequencies[bigram]/len(slavic_bigrams)
```


```python
bigrams_df.plot(kind='bar', figsize=(20,10))
plt.tick_params(axis = 'both', reset=True, labelsize=15)
plt.legend(prop={'size':20})
plt.xlabel("Bigrams", fontsize= 20, labelpad = 20)
plt.ylabel("Frequency (percent)", fontsize= 20, labelpad = 20)
plt.show()

bigrams_df
```




    <matplotlib.axes._subplots.AxesSubplot at 0x188d1a908>






    <matplotlib.legend.Legend at 0x188e61f98>






    <matplotlib.text.Text at 0x143d8b748>






    <matplotlib.text.Text at 0x188ccacc0>




![png](output_19_4.png)





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
      <th>English(BNC)</th>
      <th>Germanic(VOICE)</th>
      <th>Romance(VOICE)</th>
      <th>Slavic(VOICE)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(it, 's)</th>
      <td>0.00732095</td>
      <td>0.0111299</td>
      <td>0.0116396</td>
      <td>0.0103653</td>
    </tr>
    <tr>
      <th>(that, 's)</th>
      <td>0.0048141</td>
      <td>0.00579272</td>
      <td>0.00367629</td>
      <td>0.00358305</td>
    </tr>
    <tr>
      <th>(do, n't)</th>
      <td>0.00450116</td>
      <td>0.00383028</td>
      <td>0.00534955</td>
      <td>0.00453366</td>
    </tr>
    <tr>
      <th>(i, 'm)</th>
      <td>0.0027683</td>
      <td>0.00245656</td>
      <td>0.0032366</td>
      <td>0.00274213</td>
    </tr>
    <tr>
      <th>(i, 've)</th>
      <td>0.00193566</td>
      <td>0.000546682</td>
      <td>0.000403049</td>
      <td>0.000274213</td>
    </tr>
    <tr>
      <th>(you, 're)</th>
      <td>0.00188097</td>
      <td>0.00121952</td>
      <td>0.000879379</td>
      <td>0.00063983</td>
    </tr>
    <tr>
      <th>(there, 's)</th>
      <td>0.00181081</td>
      <td>0.000904127</td>
      <td>0.000488544</td>
      <td>0.000365617</td>
    </tr>
    <tr>
      <th>(they, 're)</th>
      <td>0.0016304</td>
      <td>0.000567708</td>
      <td>0.000537398</td>
      <td>0.000438741</td>
    </tr>
    <tr>
      <th>(he, 's)</th>
      <td>0.00162229</td>
      <td>0.000392489</td>
      <td>0.000464116</td>
      <td>0.000255932</td>
    </tr>
    <tr>
      <th>(did, n't)</th>
      <td>0.00149552</td>
      <td>0.000735918</td>
      <td>0.00118472</td>
      <td>0.000804358</td>
    </tr>
    <tr>
      <th>(you, 've)</th>
      <td>0.00138857</td>
      <td>0.000220775</td>
      <td>4.88544e-05</td>
      <td>3.65617e-05</td>
    </tr>
    <tr>
      <th>(ca, n't)</th>
      <td>0.00137492</td>
      <td>0.000620273</td>
      <td>0.000732815</td>
      <td>0.000566707</td>
    </tr>
    <tr>
      <th>(i, 'll)</th>
      <td>0.00135872</td>
      <td>0.000462577</td>
      <td>0.000464116</td>
      <td>0.000438741</td>
    </tr>
    <tr>
      <th>(we, 're)</th>
      <td>0.00114771</td>
      <td>0.000844553</td>
      <td>0.000659534</td>
      <td>0.000511864</td>
    </tr>
    <tr>
      <th>(we, 've)</th>
      <td>0.00112724</td>
      <td>0.000203253</td>
      <td>9.77087e-05</td>
      <td>0.000182809</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
