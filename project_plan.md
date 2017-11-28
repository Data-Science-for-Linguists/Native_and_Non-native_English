Katherine Kairis, kak275@pitt.edu, 10/3/2017
# Comparing English speech of native and non-native speakers
## Summary
I plan on exploring major differences between the natural speech of native and non-native English speakers, in addition to differences in speech between different L1 groups. I will use two corpora -- one containing conversations between native speakers and one containing conversations between non-native speakers.

## Data
I plan on using the Vienna-Oxford International Corpus of English (VOICE), which can be found at this link: http://ota.ox.ac.uk/desc/2542. VOICE contains conversations between speakers for various L1 backgrounds who use English as a common language. Some of the participants in this corpus are native English speakers, and since VOICE should only contain non-native speech for this project, I removed the utterances of participants who have English listed as an L1. The other corpus I use is the British National Corpus (BNC), which can be found at http://ota.ox.ac.uk/desc/2554 or http://www.natcorp.ox.ac.uk. This corpus contains the conversations and writings of British English speakers. Since this project compares speech, I will have to remove all of the written entries, which consist of around 90% of the BNC's files. Both of these corpora required modification, and both are also in XML format. Because of these modifications and the fact that I had to get used to BeautifulSoup to process the XML files, preparing the data took a fair amount of time.

## Analysis
### Completed analysis
So far I compared bigrams and hesitations between speakers in VOICE and BNC. In addition, I created three sub-corpora from VOICE -- L1 speakers of Germanic languages, L1 speakers of Romance languages, and L1 speakers of Slavic languages -- and made the same comparisons. To accomplish these analyses, I used a lot of frequency dictionaries. 
### Future plans
Some possible analyses I could perform include the following:
-	Select a few specific L1s from VOICE, and compare their participants' speech. I could make the same comparisons that I described above. Since all of these utterances will come from VOICE, all of the tags will come from the same annotation sets. I think that it could interesting to take advantage of this and possibly use part of speech tags in my analysis.
-	Use the metrics described above to compare the speech of non-native speakers conversing with several native speakers to the speech of non-native speakers conversing only with other non-native speakers.
-	Use the features I discover in my analyses to train different machine learning algorithms, and see if these algorithms can distinguish between native and non-native English speakers.
