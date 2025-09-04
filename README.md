# manx-gaelic-llm
An experimental open-source Large Language Model for the Manx Gaelic (Gaelg) language

This project aims to support and promote the revitalisation of the Manx Gaelic language and to aid in learning this language. 
It is intended as a text generation tool and, hopefully, a conversation partner with which to have very simple conversations so that learners at beginner/advanced beginner level can practice what they have learned during lessons.

After searching, it was found that no such tool exists for use in this language to date, and therefore this model could fill a gap.

Status: The project contains a tiny LLM with a BPE tokenizer and a small corpus of ~450 sentences and a vocabulary of ~250. The tiny LLM has been trained on the corpus. Gradio has been added to the code so that an interface is available to chat with the LLM. The LLM sometimes gives relevant but uncreative responses. It continues to write until it has used up all its tokens. There are too many errors and irrelevances for it to be used by learners currently. It is currently a research project and not ready for public use. 

Sources for the corpus: The corpus has been based on my own learning notes and inspired by the textbook Keim Nane and beginner lessons on the website learnmanx.com. At this stage the corpus is experimental. All data has been created by me and any sentences inspired by the above sources have been rephrased or paraphrased.

Further development: The most important task at this stage is to expand the corpus. As the project progresses, my aim is to release small demos and seek collaboration with members of the Manx-speaking community. A full, open-source pipeline is planned. 

Resources that I used to build the LLM: 
My own programming knowledge
Tutorials: Code your own Llama 4 from scratch https://www.youtube.com/watch?v=biveB0gOlak&list=WL&index=79&t=12s
Colab and Colab Gemini assistant for troubleshooting.
