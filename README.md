# Cross-domain Hate Speech Detection for Content Moderation in Greek Social Networks

This repository hosts the implementation described in our paper ["Cross-domain Hate Speech Detection for Content Moderation in Greek Social Networks"](). 

The paper introduces a state-of-the-art greek hate speech classifier, with comparable performance to Large Language Models (LLMs) in few-shot setting, for the Greek language. It also introduces a new dataset, partially annotated with data collected from Twitch to facilitate further hate speech exploration in new social networks. It further explores the efficacy of hate speech detection methods in cross-domain detection across three social networks.

We employ augmentation, balancing and transliteration techniques to deal with the unique characteristics of each dataset. Our sparse-encoder model, inspired by Switch-Transformers, outperforms all past approaches and is on par with LLMs, while being more cost effective. Our analysis shows an overall inability to generally moderate online content using single approaches, and identifies an urgent need for techniques that can work in domain agnostic scenarios (i.e., without previous knowledge of the hate speech types of interest and social networks to be applied)

# Cite as: 
N. Stylianou, T. Tsikrika, S. Vrochidis, I. Kompatsiaris, "Cross-domain Hate Speech Detection for Content Moderation in Greek Social Networks" in 23rd IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology, Bangkok, Thailand, December 9-12, 2024, IEEE.

<!-- To be added upon conference proceedings release
```bibtex
``` -->

# Acknowledgements
This work was supported by the CESAGRAM project funded by the European Union (Internal Security Fund) under Grant Agreement No. 101084974. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. The European Union cannot be held responsible for them.