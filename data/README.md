# Datasets
This directory assumes the following structure: 

* ``data/twitch`` containing the twitch dataset 
* ``data/ogtd`` containing the ogtd dataset split
* ``data/dachs`` containing the dachs dataset split

OGTD and DACHs are not provided within this repository. 

OGTD is available online, as used in this dataset. 

DACHS is available online using tweet ids. Your effective dataset may vary depending on the time of collection. 

Twitch (``Twitch_hatespeech_greek.zip``) is provided in the following settings: 

1. Annotated dataset: ``data/twitch/annotations/test.json``
    
    Contains the 200 annotated messages. 
    
    Labels: 

        0 - Not Hate Speech
        1 - Hatespeech

2. Twitch dataset (annotated and clean): ``data/twitch/twitch_data.json``
    
    Contains 6601 messages(including the 200 annotated)

3. Per user messages (pseudonimzed usernames): ``data/twitch/per_user/*.json``
    
    Contains the 6601 messages, split _per user_.


