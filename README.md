---
title: manx-gaelic-llm
emoji: "\U0001F1EE\U0001F1F2"
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# manx-gaelic-llm
An experimental open-source Large Language Model for the Manx (Gaelg) language

This project aims to support and promote the revitalisation of the Manx language and to aid in learning this language. 
It is intended as a text generation tool and, as the project progresses, a conversation partner with which to have very simple conversations so that learners at beginner/advanced beginner level can practice what they have learned during lessons.

After searching, it was found that no such tool exists for use in this language to date, and therefore this model could fill a gap.

Status: The project contains a tiny LLM with a BPE tokenizer and a small corpus of ~1000 sentences and a vocabulary of ~300. The tiny LLM has been trained on the corpus. Gradio has been added to the code so that an interface is available to chat with the LLM. The LLM sometimes gives relevant but uncreative responses. It doesn't answer the majority of questions well or at all, but it is able to generate some sensible sentences when given the first one or two words of a sentence. At the moment it could be best used as a sentence generator at beginner level once accuracy is improved a little more. It is currently a research project and not ready for public use. 

Sources for the corpus: The corpus has been based on my own learning notes and inspired by the textbook Loayr Gaelg! Keim Nane and beginner lessons on the website learnmanx.com. At this stage the corpus is experimental. It only includes vocabulary and grammar structures taught at Level 1 (Keim Nane) and is intended as an aid for learners at this level. As the corpus grows, I plan to expand it to include language taught at Level 2 (Keim Jees) All data has been created by me and any sentences inspired by the above sources have been rephrased or paraphrased. However, I hope for halp from and collaboration with more fluent Manx speakers as the project grows. I have written and curated the corpus myself so that it contains the specific vocabulary and grammar structures that student learn at certain levels, and so that it is not contaminated with words or phrases from other languages (unless commonly used in Manx). The LLM is intended to be monolingual.

Further development: The most important task at this stage is to expand the corpus. As the project progresses, my aim is to release small demos and seek collaboration with members of the Manx-speaking community. In addition to this, I am focusing on steps to expand and improve the transformer. A full, open-source pipeline is planned. 

As well as my own learning notes, I used the following language resources as inspiration and as aids to help me create a corpus:
* Loayr Gaelg! Keim Nane
* Cowag Nane
* https://dictionaryq.com/gaelg/
* https://www.learnmanx.com/learning/beginner/
* https://corpus.gaelg.im/

Here is a list of some resources and aids which have been useful in helping me with ideas on how to best build the transformer (I built this transformer from scratch and am continually looking at ways to improve it):
* Colab (plus colab AI assistant to help debug and test)
* https://www.youtube.com/watch?v=biveB0gOlak&list=WL&index=76&t=12s (the first version of this model was a tiny llama-type model based on the model built in this tutorial.)
* https://www.youtube.com/watch?v=5avSMc79V-w
* https://www.youtube.com/watch?v=UU1WVnMk4E8&t=5s
* https://www.youtube.com/watch?v=p3sij8QzONQ&list=WL&index=94
* https://course.fast.ai/
