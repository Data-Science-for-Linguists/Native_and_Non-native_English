
Katherine Kairis, kak275@pitt.edu, 12/15/2017

# Processing Data for VOICE


```python
from bs4 import BeautifulSoup
import glob
import re
import nltk

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

## Reading in data from the corpus
The first step is reading in the contents of each file in the corpus. As the files are read in, two different dictionaries are populated: conversations and participants.  
Since the files are in XML format, conversations contains BeautifulSoup objects for each file's contents.  
participants contains information(native language, age, occupation, etc.) about every participant in the corpus

### Helper functions: participant_info and conversation_lines
* participant_info takes the contents of one conversation and a dictionary containing participants' information. It uses BeautifulSoup commands to get all of the participants listed in the conversation, in addition to the person's native language(s), sex, age, occupation, and role. All of this information is stored in a subdictionary, which are the values of the participants dictionary.  
* conversation_lines takes a file path, a single conversation's XML contents, and a dictionary of conversations. The file name is saved in the dictionary as a key, and the XML contents are saved in the dictionary as the file's value.


```python
#Takes a conversation (in XML format) and a participants dictionary. Gets the participants in the
#conversation their information (native language, age, occupation, etc.), and adds them
#to the participants dictionary.

def participant_info(contents, participants_dict):
    #Get all of the participants in the given conversation, and iterate through each person in the list
    people = contents.find('listPerson', {'type': 'identified'}).findAll('person')
    for p in people:
        #info is a subdirectory that contains a single participant's information. The participant's 
        #id will be added to the participants dictionary, and the info dictionary will be the value
        #corresponding to the participant.
        info = {}
        
        name = p['xml:id']
        c = name.split("_")[0]
        c = c + ".xml"
        
        #Get the current spekers's role, age, sex, and the conversation that they participate in.
        info['conversation'] = c
        info['role'] = p['role']
        info['age'] = p.age.get_text()
        info['sex'] = p.sex.get_text()
        
        #In some cases, the occupation isn't listed. If it is included, get the text of the occupation field.
        #If it isn't included, "None" will be stored as the occupation, since p.occupation returns "None."
        try:
            info['occupation'] = p.occupation.get_text()
        except AttributeError:
            info['occupation'] = p.occupation
        
        #Get a list of the languages that the participant speaks. Iterate through the list, and add them to the
        #dictionary according to the speaker's level (ie. L1).
        languages = p.findAll('langKnown')
        for l in languages:
            level = l['level']
            language = l['tag']
            language = language.split('-')[0]
        
            if level in info:
                info[level].append(language)
            else:
                info[level] = [language]
    
        #Get the participant's ID number, and make it a key in the participants dictionary. The value will be
        #the info dictionary
        participants_dict[name] = info
```


```python
#Takes a file's path, its XML contents, and a dictionary. Create a dictionary whose keys are the 
#file names, and whose values are the files'/conversations' XML contents
def conversation_lines(file, contents, conversation_dict):
    #Get only the file name from the file's path
    file_name = file.split("/")[-1]
    #Add the file name as a key in the conversation dictionary. Its value is its XML contents
    conversation_dict[file_name] = contents
```

### Reading and initial processing of the conversation files


```python
#Get the XML files containing transcripts of the conversations.
#Delete the first file, which is a header file
transcripts = glob.glob('data/VOICE/VOICE2.0XML/XML/*.xml')
del transcripts[0]

tagged_transcripts = glob.glob('data/VOICE/VOICEPOSXML2.0/XML/*.xml')
del transcripts[0]
```


```python
#Read in the XML files from the corpus, and process each file as it's read in.
conversations = {}
participants = {}

#For each transcript/conversation in the corpus, get the XML text and the information about the participants
for t in transcripts:
    #Open the file
    file = open(t, 'r')
    text = file.read()
    xml_contents = BeautifulSoup(text, 'xml')
    
    #Add the XML contents from the files to the conversations dictionary.
    #The dictionary keys are the file names, and their values are the file's XML text.
    conversation_lines(t, xml_contents, conversations)
    
    #Use the participant_info function to get information about the participants in each conversation
    #The keys of the dictionary are the participants' ids, and the values are sub-dictionaries
    #containing their information.
    participant_info(xml_contents, participants)
    
    file.close()
```

### Results of initial processing


```python
#There are 150 conversations in VOICE. 
#Show the names of the first 10 conversations/files in the conversations dictionary. 
len(conversations.keys())
list(conversations.keys())[:10]
```




    150






    ['EDcon4.xml',
     'EDcon496.xml',
     'EDcon521.xml',
     'EDint328.xml',
     'EDint330.xml',
     'EDint331.xml',
     'EDint604.xml',
     'EDint605.xml',
     'EDsed251.xml',
     'EDsed301.xml']




```python
#The values in the conversations dictionary are BeautifulSoup objects/XML contents. 
type(conversations['EDsed301.xml'])

#Show an utterance in one of the files/conversations 
conversations['EDsed301.xml'].find('u')
```




    bs4.BeautifulSoup






    <u who="#EDsed301_S1" xml:id="EDsed301_u_1"><unclear> excuse me </unclear> cou- could we talk about the tests<c function="rise" type="intonation"/><pause/></u>




```python
#There are 1253 participants in VOICE corpus.
len(participants)

#Show the first 10 participants in the participants dictionary
list(participants.keys())[:10]
```




    1253






    ['EDcon4_S4',
     'EDcon4_S1',
     'EDcon4_S2',
     'EDcon4_S5',
     'EDcon4_S6',
     'EDcon4_S7',
     'EDcon4_S8',
     'EDcon4_S3',
     'EDcon4_S9',
     'EDcon496_S4']




```python
#Show some entries (information about the participants) in the participants dictionary
participants['EDsed31_S14']
participants['POcon549_S11']
participants['EDsed362_S14']
participants['EDsed362_S14']['L1']
```




    {'L1': ['fre'],
     'age': '17-24',
     'conversation': 'EDsed31.xml',
     'occupation': None,
     'role': 'student',
     'sex': 'female'}






    {'L1': ['nor'],
     'age': '35-49',
     'conversation': 'POcon549.xml',
     'occupation': 'representative of European higher education network',
     'role': 'participant',
     'sex': 'male'}






    {'L1': ['eng', 'heb', 'dut'],
     'age': '17-24',
     'conversation': 'EDsed362.xml',
     'occupation': 'student',
     'role': 'participant',
     'sex': 'female'}






    ['eng', 'heb', 'dut']



## Processing and refining the data
The first step involves making lists of the native speakers and of the bilingual participants. These lists will be used to ensure that native speakers and bilinguals are not included in the final data. 
At this point, all of the conversations are in XML format, so BeautifulSoup is used to determine which utterances shouldn't be included in the data because of certain tags (unclear, non-English speech, etc.), and to get the text of the utterances. The text is then used to create dictionaries of tokenized conversations and (word, tag) pairs.

### Getting the native speakers


```python
#Get native English speakers
native_speakers = []

#There are multiple ways that English is listed as an L1: 
#("eng", "eng-US", "eng-CA", "eng-GB", "eng-GY", "eng-AU", etc)

#Use a regular expression to find all of these instances
r = re.compile("eng.*")

for person in participants:
    #returns a list of all languages that contain "eng.*" The length of this list should be 1 or 0. If it's 1, the
    #participant has English listed as an L1.
    english = list(filter(r.match, participants[person]['L1']))
    
    if len(english) != 0:
        #print(person, ':', participants[person])
        native_speakers.append(person)
```


```python
#There are 86 native speakers in the corpus
len(native_speakers)

#Show the first 10 native speakers in the list
native_speakers[:10]
```




    86






    ['EDcon496_S2',
     'EDcon521_S1',
     'EDint328_S3',
     'EDint330_S2',
     'EDint330_S4',
     'EDsed251_S3',
     'EDsed301_S6',
     'EDsed362_S1',
     'EDsed362_S11',
     'EDsed362_S14']




```python
#Show information about two of the native speakers
participants['EDsed301_S6']
participants['EDsed362_S1']
```




    {'L1': ['eng'],
     'age': '17-24',
     'conversation': 'EDsed301.xml',
     'occupation': 'student',
     'role': 'student',
     'sex': 'female'}






    {'L1': ['eng'],
     'age': '50+',
     'conversation': 'EDsed362.xml',
     'occupation': 'professor of history',
     'role': 'teacher',
     'sex': 'male'}



### Getting bilingual participants


```python
bilinguals = []

#Iterate through each participant in the participants dictionary.
for p in participants:
    #Get the participant's L1(s)
    languages = participants[p]['L1']
    
    #If the participant has more than one L1 listed, add them to the list of bilinguals
    if len(languages) > 1:
        bilinguals.append(p)
```


```python
#There are 43 bilinguals in the corpus
len(bilinguals)

#Show the first 10 bilinguals in the list
bilinguals[:10]
```




    43






    ['EDcon496_S2',
     'EDcon521_S1',
     'EDint328_S3',
     'EDint330_S2',
     'EDint330_S4',
     'EDint604_S3',
     'EDsed251_S3',
     'EDsed251_S9',
     'EDsed31_S6',
     'EDsed362_S11']




```python
participants['EDsed31_S6']
```




    {'L1': ['ger', 'ind'],
     'age': '17-24',
     'conversation': 'EDsed31.xml',
     'occupation': 'student of medicine',
     'role': 'student',
     'sex': 'male'}



### Helper function: valid_utterance
The valid_utterance function is used when creating dictionaries for both tokens and (word, tag) pairs. It takes a single utterance, and ensures that it can be included in the data. An utterance cannot be spoken by a native speaker or a bilingual, include non-English or unclear speech, or involve the speaker reading out loud.


```python
#Checks to make sure the line can be added to the dictionary.
#A line must meet the following criteria: 
#the speaker cannot be a native speaker of English (not handled in this function)
#the speaker cannot be bilingual
#the speaker must be listed in the participant directory
#the line cannot contain any non-English speech
#the line cannot contain any unclear speech
#the line cannot contain the speaker reading anything out loud
def valid_utterance(participant, line):
    if participant in bilinguals:
        return False
    if participant not in participants:
        return False
    if line.foreign != None:
        return False
    if line.unclear != None: 
        return False
    if line.reading_aloud != None:
        return False
    if line.reading != None :
        return False
    return True
```

### Creating a dictionary of tokenized conversations


```python
#Iterate through all of the files in VOICE to create a nested dictionary that contains the word tokens of the
#conversations.
#The keys of the dictionaries are the file names. The values of these entries are subdictionaries. The keys of the
#subdictionary are (line_number, participant) tuples, and the values are lists of tokens.
tokenized_conversations = {}
conversation_info = {}

#Iterate through each conversation's contents
for file in conversations:
    #Conv_lines contains all of the utterances for the current, single conversation
    conv_lines = {}
    
    #At this point, the contents of the current conversation is still in XML/Beautifuloup, and
    #include a lot of extra information.
    #In the corpus, lines that are tagged with 'u' indicate an utterance. To get all of the utterances
    #in the current file, use find.All('u')
    c = conversations[file].findAll('u')
    
    #Iterate through all of the utterance in the file
    for l in c:
        #Get the speaker and the id of the current utterance
        participant = l['who'].replace("#", "")
        line_id = l['xml:id']
        #Get the text of the current utterance (remove XML tags), and tokenize the utterance using NLTK
        text = l.get_text()
        tokens = nltk.word_tokenize(text)
        
        #If the participant is a native speaker, don't include them in the dictionary of tokenized conversations
        if participant in native_speakers:
            continue
        
        #Otherwise, check if the utterance is valid using the valid_utterance function. If it is valid,
        #add it to conv_lines. The key is a tuple consisting of (line number, speaker), and the 
        #value is the tokenized speech.
        elif len(text) != 0 and valid_utterance(participant, l) == True:
            key = (line_id, participant)
            conv_lines[key] = tokens
    
    #When done iterating through all of the utterances of the current conversation, 
    #add it to the dictionary of tokenized conversations. The key is the current 
    #file/conversation, and the value conv_lines.
    tokenized_conversations[file] = conv_lines
```


```python
#There are 150 conversations
len(list(tokenized_conversations.keys()))

first_entry = list(tokenized_conversations.keys())[0]
first_conversation = tokenized_conversations[first_entry]

#There are 671 utterances in the first conversation
len(list(first_conversation.keys()))
```




    150






    671




```python
#Show the speaker and line number of the first 10 utterances in the first conversation
list(first_conversation.keys())[:10]

#Show a few utterances in the first conversation
tokenized_conversations[first_entry][('EDcon4_u_5', 'EDcon4_S1')]
tokenized_conversations[first_entry][('EDcon4_u_6', 'EDcon4_S3')]
```




    [('EDcon4_u_2', 'EDcon4_S2'),
     ('EDcon4_u_3', 'EDcon4_S1'),
     ('EDcon4_u_4', 'EDcon4_S2'),
     ('EDcon4_u_5', 'EDcon4_S1'),
     ('EDcon4_u_6', 'EDcon4_S3'),
     ('EDcon4_u_7', 'EDcon4_S1'),
     ('EDcon4_u_9', 'EDcon4_S1'),
     ('EDcon4_u_11', 'EDcon4_S1'),
     ('EDcon4_u_12', 'EDcon4_S2'),
     ('EDcon4_u_13', 'EDcon4_S1')]






    ['and', 'and', 'er']






    ['oh', 'sorry']



### Creating a dictionary of part-of-speech tagged conversations


```python
#Get the text from the part of speech-tagged files
tagged_conv_lines = {}
for t in tagged_transcripts:
    file = open(t, 'r')
    text = file.read()
    xml_contents = BeautifulSoup(text, 'xml')
    #Use conversation lines to get a dictionary of the XML contents of each conversation
    conversation_lines(t, xml_contents, tagged_conv_lines)
    file.close()
```


```python
#Iterate through all of the files in VOICE to a nested dictionary that contains the (word, tag) tuples from the
#conversations.
#The keys of the dictionaries are the file names. The values of these entries are subdictionaries. The keys of the
#subdictionary are (participant, line_number) tuples, and the values are lists of (word, tag) tuples.
tagged_conversations = {}
tagged_conversations_native = {}

#Iterate through each conversation's contents
for file in tagged_conv_lines:
    #Conv_lines contains all of the utterances for the current, single conversation
    conv_lines = {}
    #Create a second dictionary for native speakers. It will work the same way as conv_lines
    conv_lines_native = {}
    
    #In the corpus, lines that are tagged with 'u' indicate an utterance. To get all of the utterances
    #in the current file, use find.All('u')
    c = tagged_conv_lines[file].findAll('u')
    
    #Iterate through all of the utterances in the file
    for l in c:
        #Utterance and native_utterance contain the (word,tag) pairs of a single utterance(the 
        #current one in the iteration)
        utterance = []
        native_utterance = []
        
        #Get the speaker and the id of the current utterance
        participant = l['who'].replace("#", "")
        line_id = l['xml:id']
        key = (line_id, participant)
        
        #Check if the current utterance is valid using the valid_utterance function. If it is valid,
        #add it to either conv_lines or conv_lines_native.
        if valid_utterance(participant, l) == True:  
            #In the part-of-speech tagged files, the tag 'w' indicates a word, and includes
            #the words part of speech and lemma. To get all of the word, use findall('w')
            tags = l.findAll('w')
            
            #Iterate through all of the words/tags in the utterance
            for t in tags:
                #Get the word and its part of speech
                word = t.text
                ana = str(t).split()[1]
                ana = ana.split("=")
                tag = ana[1][2:]
                tag = tag.split('"')[0]
                
                #Add the word,tag pair to either native_utterance and conv_lines_native (if the participant
                #is in the list of native speakers) or to utterance and conv_lines
                if participant in native_speakers:
                    native_utterance.append((word, tag))
                    conv_lines_native[key] = native_utterance
                    
                else:
                    utterance.append((word, tag))  
                    conv_lines[key] = utterance
    
    #When done iterating through all of the utterances, update both tagged conversations dictionaries
    #with the subdictionaries containing tagged utterances from the current conversation
    tagged_conversations[file] = conv_lines
    tagged_conversations_native[file] = conv_lines_native
```


```python
entry = list(tagged_conversations.keys())[5]
example_conversation = tokenized_conversations[entry]

#There are 448 utterances in the example conversation
len(list(example_conversation.keys()))

#Show the line numbers and speakers of the first 10 utterances in the example conversation
list(example_conversation.keys())[:10]
```




    448






    [('EDint328_u_1', 'EDint328_S1'),
     ('EDint328_u_3', 'EDint328_S2'),
     ('EDint328_u_4', 'EDint328_S1'),
     ('EDint328_u_6', 'EDint328_S2'),
     ('EDint328_u_7', 'EDint328_S1'),
     ('EDint328_u_8', 'EDint328_S2'),
     ('EDint328_u_9', 'EDint328_S1'),
     ('EDint328_u_10', 'EDint328_S1'),
     ('EDint328_u_12', 'EDint328_S1'),
     ('EDint328_u_14', 'EDint328_S1')]




```python
#Show some tagged utterances in the example conversation
tagged_conversations[entry][('EDint328_u_1', 'EDint328_S1')]
tagged_conversations[entry][('EDint328_u_10', 'EDint328_S1')]
```




    [('no', 'REfRE'),
     ('no', 'REfRE'),
     ('un-', 'XXfXX'),
     ('unfortunately', 'RBfRB'),
     ('_1', 'PAfPA'),
     ('okay', 'REfRE'),
     ('okay', 'REfRE'),
     ('_0', 'PAfPA')]






    [('no', 'DTfDT'), ('problem', 'NNfNN'), ('_0', 'PAfPA')]



## Pickling the data


```python
#Save the two dictionaries as pickle files
import pickle
```


```python
f = open('VOICE_tokenized.p', 'wb')
pickle.dump(tokenized_conversations, f, -1)
f.close()
```


```python
f = open('VOICE_tagged.p', 'wb')
pickle.dump(tagged_conversations, f, -1)
f.close()
```


```python
f = open('VOICE_native_tagged.p', 'wb')
pickle.dump(tagged_conversations_native, f, -1)
f.close()
```


```python
f = open("VOICE_participant_info.p", 'wb')
pickle.dump(participants, f, -1)
f.close()
```
