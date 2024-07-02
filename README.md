Hi Jiameng,

In this repository you can find my attempt at the take home task provided following our introductory call.
There are two main pieces of code, the interactive and static parts. I used whisper for ASR, Mistral 7B as the LLM and MozillaTTS for TTS model.

I used the static part of the code as a way to test the models were working as intended, I downloaded a librispeech dataset from https://www.openslr.org/12 and used those audio files as input audio. 
To run the static code, you can find it in: Code/Static/FullStackStatic.py

I then attempted to create an "interactive" session where you talk interactively with the LLM, I feel like this was a mistake, as I ended up wasting a lot of time getting everything to work.
Although it took me longer than expected, I am happy with the results in the interactive session. The responses from the LLM are not the best, but I think I wasted too much time trying to improve its responses.
To run the interactive code, you can find it in: Code/Interactive/FullStackInteractive.py

I attach all python packages I used just as I acquired from the venv environment I was using, I hope there are no dependencies issues. Hopefully you can run the setup_env.sh in EnvSetup folder, allocate the new environment as the interpreter and everything works fine, like it did for me.

One thing to watch out for is HuggingFace tokens, as I used Mistral7B for my LLM, for safety reasons I took my tokens out before uploading my code, I hope this is not too big of an inconvenience.

I apologise for not attempting more extended tasks, but due to heavy University work, this is the best I could do :)

Thanks for this opportunity!
Hope to hear from you soon!
