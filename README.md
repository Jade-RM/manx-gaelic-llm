---
title: manx-gaelic-llm
emoji: 🚀
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: tiny_llm_with_gradio_interface.py
pinned: false
---

# manx-gaelic-llm
An experimental open-source Large Language Model for the Manx (Gaelg) language

This project aims to support and promote the revitalisation of the Manx language and to aid in learning this language. 
It is intended as a text generation tool and, as the project progresses, a conversation partner with which to have very simple conversations so that learners at beginner/advanced beginner level can practice what they have learned during lessons.

After searching, it was found that no such tool exists for use in this language to date, and therefore this model could fill a gap.

Status: The project contains a tiny LLM with a BPE tokenizer and a small corpus of ~1400 sentences and a vocabulary of ~400. The tiny LLM has been trained on the corpus. Gradio has been added to the code so that an interface is available to chat with the LLM. The LLM gives relevant responses most of the time when asked simple questions or presented with simple statements. Sometimes it answers creatively (not sentences lifted directly from the corpus) and it has also asked simple questions a couple of times during conversations. While earlier it worked as a sentence generator, it is no longer able to complete sentences. It does, however, now function as a simple conversation partner. It can discuss topics such as likes and dislikes, feelings (e.g., ta mee skee, ta mee maynrey...), simple daily activities (e.g., ta mee roie, ta mee gobbragh...), simple activities in the past (e.g., ren mee gobbragh jea), and basic information such as where one lives, what pets one has etc. It is currently a research project and not ready for public use, but with expanded data it could soon become useful as a tool for beginner conversation practice. 

Sources for the corpus: The corpus has been based on my own learning notes and inspired by the textbook Loayr Gaelg! Keim Nane and beginner lessons on the website learnmanx.com. At this stage the corpus is experimental. It only includes vocabulary and grammar structures taught at Level 1 (Keim Nane) and is intended as an aid for learners at this level. As the corpus grows, I plan to expand it to include language taught at Level 2 (Keim Jees) All data has been created by me and any sentences inspired by the above sources have been rephrased or paraphrased. However, I hope for help from and collaboration with more fluent Manx speakers as the project grows. I have written and curated the corpus myself so that it contains the specific vocabulary and grammar structures that student learn at certain levels, and so that it is not contaminated with words or phrases from other languages (unless commonly used in Manx). The LLM is intended to be monolingual.

Further development: The most important task at this stage is to expand the corpus at Keim Nane level so that the LLM becomes more conversational, more creative and more accurate. As the project progresses, my aim is to release small demos and seek collaboration with members of the Manx-speaking community. In addition to this, I am focusing on steps to expand and improve the transformer. A space has been created for the demo model on Hugging Face. A full, open-source pipeline is planned. 

As well as my own learning notes, I used the following language resources as inspiration and as aids to help me create a corpus:
* Keim Nane Level One of Coorse Gaelgagh cour sleih aasit A Manx course for adults Loayr Gaelg! Speak Manx! (n.d.). Retrieved November 11, 2025, from https://www.learnmanx.com/media//news%20pictures/New%202022%20pictures/FINAL%20Loayr%20Gaelg%20Keim%201.pdf
* Cowag A series of conversational pieces and additional material, with accompanying CD, for the learner of Manx Gaelic Lioar Nane-Book One. (n.d.). Retrieved November 11, 2025, from https://www.learnmanx.com/media//cowag_1/cowag%2001.pdf
* Fockleyreen: Manx - English Dictionary. (2025). Dictionaryq.com. https://dictionaryq.com/gaelg/‌
* Beginner Lessons | Learn Manx. (2025). Learnmanx.com. https://www.learnmanx.com/learning/beginner/
* Manx Corpus Search. (2025). Gaelg.im. https://corpus.gaelg.im/

Here is a list of some resources and aids which have been useful in helping me with ideas on how to best build the transformer (I built this transformer from scratch and am continually looking at ways to improve it):
* Colab (plus colab AI assistant to help debug and test)
* freeCodeCamp.org. (2025, April 24). Code Your Own Llama 4 LLM from Scratch – Full Course. YouTube. https://www.youtube.com/watch?v=biveB0gOlak (the first version of this model was a tiny llama-type model based on the model built in this tutorial.)
* freeCodeCamp.org. (2025, April 1). Code DeepSeek V3 From Scratch in Python - Full Course. YouTube. https://www.youtube.com/watch?v=5avSMc79V-w
* Create a Large Language Model from Scratch with Python – Tutorial. (n.d.). Www.youtube.com. https://www.youtube.com/watch?v=UU1WVnMk4E8
* freeCodeCamp.org. (2025, September 23). LLMs from Scratch – Practical Engineering from Base Model to PPO RLHF. YouTube. https://www.youtube.com/watch?v=p3sij8QzONQ
* Practical Deep Learning for Coders | Practical Deep Learning for Coders. (n.d.). Course.fast.ai. https://course.fast.ai/
