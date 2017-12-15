
Katherine Kairis, kak275@pitt.edu, 12/15/2017

# Processing Data for BNC


```python
import os
import glob
from bs4 import BeautifulSoup

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
file = open('data/BNC/download/Texts/G/G6/G62.xml', 'r')
text = file.read()
spoken = BeautifulSoup(text, 'xml')
file.close()
```

## Get the spoken files


```python
#The BNC has both transcripts of speech and written entries. Spoken text is differentiated from written text with the 
#<stext> tag, which includes all of the speech in the file.
spoken_lines = spoken.stext
```


```python
#The files are organized in several nested directories. Get all of the files from all of the directories.
spoken_files = []
root = 'data/BNC/download/Texts/'
for path, subdirs, files in os.walk(root):
    for name in files:
        file_name = os.path.join(path, name)

        if file_name.endswith('.xml'):
            file = open(file_name, 'r')
            text = file.read()
        
            if('<stext') in text and file_name not in spoken_files:
                spoken_files.append(file_name)
            
            file.close()
```

    data/BNC/download/Texts/K/KS/KSW.xml


## Test processing one conversation


```python
#Try getting the words and tags of the last file in the corpus, text.
#file = open('data/BNC/download/Texts/H/HU/HUE.xml', 'r')
file = open('data/BNC/download/Texts/K/KS/KSW.xml', 'r')
text = file.read()
xml_contents = BeautifulSoup(text, 'xml')
file.close()
```


```python
#Show a segment of the file's XML 
text[:1000]
```




    '<bncDoc xml:id="KSW"><teiHeader><fileDesc><titleStmt><title>  6 conversations recorded by `Richard4\' (PS6SG) [dates unknown] with 5 interlocutors, totalling 1098 s-units, 6020 words (duration not recorded). </title><respStmt><resp> Data capture and transcription </resp><name> Longman ELT </name> </respStmt></titleStmt><editionStmt><edition>BNC XML Edition, December 2006</edition></editionStmt><extent> 6172 tokens; 6467 w-units; 1099 s-units </extent><publicationStmt><distributor>Distributed under licence by Oxford University Computing Services on behalf of the BNC Consortium.</distributor><availability> This material is protected by international copyright laws and may not be copied or redistributed in any way. Consult the BNC Web Site at http://www.natcorp.ox.ac.uk for full licencing and distribution conditions.</availability><idno type="bnc">KSW</idno><idno type="old"> XRi6SG </idno></publicationStmt><sourceDesc><recordingStmt><recording xml:id="KSWRE000" n="135301" type="Walkman"/><'




```python
#Gets all utterances in the file
utterances = xml_contents.findAll('u')
len(utterances)
```




    759




```python
#Gets the first utterance in the file
u1 = utterances[0]
u1
```




    <u who="KSWPSUNK">
    <s n="1"><w c5="AV0" hw="right" pos="ADV">Right</w><c c5="PUN">, </c><w c5="ITJ" hw="hello" pos="INTERJ">hello</w><c c5="PUN">, </c><w c5="ITJ" hw="yeah" pos="INTERJ">yeah</w><c c5="PUN">, </c><w c5="PNP" hw="we" pos="PRON">we</w><w c5="VBB" hw="be" pos="VERB">'re </w><w c5="AVP" hw="back" pos="ADV">back</w><c c5="PUN">.</c></s>
    <s n="2"><w c5="UNC" hw="erm" pos="UNC">Erm  </w><pause/><w c5="NN1-VVB" hw="miss" pos="SUBST">Miss </w><w c5="VBZ" hw="be" pos="VERB">is </w><w c5="VVG" hw="try" pos="VERB">trying </w><w c5="TO0" hw="to" pos="PREP">to </w><w c5="VVI" hw="get" pos="VERB">get </w><w c5="VVN" hw="rid" pos="VERB">rid </w><w c5="PRF" hw="of" pos="PREP">of </w><w c5="PNP" hw="we" pos="PRON">us</w><c c5="PUN">.</c></s>
    <s n="3"><w c5="UNC" hw="erm" pos="UNC">Erm  </w><pause/><w c5="PNP" hw="we" pos="PRON">we</w><w c5="VBB" hw="be" pos="VERB">'re </w><w c5="VVG" hw="sunbthe" pos="VERB">sunbathing  </w><pause/><mw c5="CJS"><w c5="AV0" hw="even" pos="ADV">even </w><w c5="CJS" hw="though" pos="CONJ">though </w></mw><w c5="EX0" hw="there" pos="PRON">there </w><w c5="UNC" hw="ai" pos="UNC">ai</w><w c5="XX0" hw="not" pos="ADV">n't </w><w c5="AT0" hw="no" pos="ART">no </w><w c5="NN1" hw="sun" pos="SUBST">sun </w><w c5="TO0" hw="to" pos="PREP">to </w><w c5="VVI" hw="sunbathe" pos="VERB">sunbathe </w><w c5="PRP" hw="at" pos="PREP">at</w><c c5="PUN">.</c></s></u>




```python
#There are three sentences in the first utterance
sents = u1.findAll('s')
len(sents)

#Show the first sentence in the utterance
sents[0] #XML
sents[0].text #Text
```




    3






    <s n="1"><w c5="AV0" hw="right" pos="ADV">Right</w><c c5="PUN">, </c><w c5="ITJ" hw="hello" pos="INTERJ">hello</w><c c5="PUN">, </c><w c5="ITJ" hw="yeah" pos="INTERJ">yeah</w><c c5="PUN">, </c><w c5="PNP" hw="we" pos="PRON">we</w><w c5="VBB" hw="be" pos="VERB">'re </w><w c5="AVP" hw="back" pos="ADV">back</w><c c5="PUN">.</c></s>






    "Right, hello, yeah, we're back."




```python
#Gets speaker of the first utterance
u1['who']
```




    'KSWPSUNK'




```python
#Gets the first sentence of the utterance (in XML format)
segment1 = utterances[0].findAll('s')[0]
```


```python
segment1
```




    <s n="1"><w c5="AV0" hw="right" pos="ADV">Right</w><c c5="PUN">, </c><w c5="ITJ" hw="hello" pos="INTERJ">hello</w><c c5="PUN">, </c><w c5="ITJ" hw="yeah" pos="INTERJ">yeah</w><c c5="PUN">, </c><w c5="PNP" hw="we" pos="PRON">we</w><w c5="VBB" hw="be" pos="VERB">'re </w><w c5="AVP" hw="back" pos="ADV">back</w><c c5="PUN">.</c></s>




```python
#Gets the text of the first sentence
segment1.text
```




    "Right, hello, yeah, we're back."




```python
#Get all of the words (in XML format) of the sentence
words = segment1.findAll('w')
```


```python
words
```




    [<w c5="AV0" hw="right" pos="ADV">Right</w>,
     <w c5="ITJ" hw="hello" pos="INTERJ">hello</w>,
     <w c5="ITJ" hw="yeah" pos="INTERJ">yeah</w>,
     <w c5="PNP" hw="we" pos="PRON">we</w>,
     <w c5="VBB" hw="be" pos="VERB">'re </w>,
     <w c5="AVP" hw="back" pos="ADV">back</w>]




```python
#Get the word and tag of each word entry
for w in words:
    print(w.text, w['pos'])
```

    Right ADV
    hello INTERJ
    yeah INTERJ
    we PRON
    're  VERB
    back ADV


## Processing the entire BNC corpus


```python
#Iterate through all of the files in the BNC to two nested dictionaries: one containing word tokens of the
#conversations, and the other containing (word, tag) tuples for every word in the corpus.
#The keys of the dictionaries are the file names. The values of these entries are subdictionaries. The keys of the
#subdictionary are (participant, line_number) tuples, and the values are either lists of tokens or (word, tag) tuples.
tokenize = {}
tagged = {}

for file_name in spoken_files:
    token_dict = {}
    tag_dict = {}
        
    file = open(file_name, 'r')
    text = file.read()
    xml_contents = BeautifulSoup(text, 'xml')
    
    file.close()
    
    #The contents are still in XML, so use BeautifulSoup commands. The tag 'u' indicates that the line
    #is an utterance. To get all utterances, use findAll('u')
    utterances = xml_contents.findAll('u')
    
    #Iterate through all of the utterances in the current conversation
    for u in utterances:
        #Get the speaker and all of the sentences in the current utterance. Sentences are are marked
        #with 's', so use findAll('s') to get all of the sentences in the utterance.
        speaker = u['who']
        segments = u.findAll('s')
        
        #Iterate through all of the sentences in the utterance
        for s in segments:
            #Get the line number of the sentence
            line = s['n']
            #Get the words in the sentence
            words = s.findAll('w')
            
            token_list = []
            tag_list = []
            
            #Iterate through all of the word elements in the sentence, and get the word's text
            #and part of speech tag.
            for w in words:
                word = w.text
                word = word.lower()
                word = word.replace(" ", "")
                tag = w['pos']
                
                #Add the word to the list of tokens. Add a tuple containing the word and its
                #part of speech to the list of tags
                token_list.append(word)
                tag_list.append((word, tag))
            
            #When done iterating through the sentence, add it to the token and tag dictionaries.
            #The keys are tuples containing the speaker and the line, and the values are the lists
            #of tokens and tags
            token_dict[(speaker, line)] = token_list
            tag_dict[(speaker, line)] = tag_list
            
    #After iterating through all of the utterances in the conversation, add the utterances to
    #the dictionary. The key is the name of the file, and the value is the dictionary containing
    #sentences' words or (word, tag) pairs
    tokenize[file_name] = token_dict
    tagged[file_name] = tag_dict
```

## Picking the data


```python
#Save the dictionaries as pickle files.
import pickle
```


```python
f = open('BNC_tokenized.p', 'wb')
pickle.dump(tokenize, f, -1)
f.close()
```


```python
f = open('BNC_tagged.p', 'wb')
pickle.dump(tagged, f, -1)
f.close()
```
