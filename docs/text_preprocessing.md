# Text Pre-Processing

[retriv](https://github.com/AmenRa/retriv) provides several resources for multi-lingual text pre-processing, aiming to maximize its retrieval effectiveness.

## Stemmers
[Stemmers](https://en.wikipedia.org/wiki/Stemming) reduce words to their word stem, base or root form.  
[retriv](https://github.com/AmenRa/retriv) supports the following stemmers:
- [snowball](https://snowballstem.org) (default)  
The following languages are supported by Snowball Stemmer: 
Arabic, Basque, Catalan, Danish, Dutch, English, Finnish, French, German, Greek, Hindi, Hungarian, Indonesian, Irish, Italian, Lithuanian, Nepali, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Tamil, Turkish.  
To select your preferred language simply use `<language>` .
- [arlstem](https://www.nltk.org/api/nltk.stem.arlstem.html) (Arabic)
- [arlstem2](https://www.nltk.org/api/nltk.stem.arlstem2.html) (Arabic)
- [cistem](https://www.nltk.org/api/nltk.stem.cistem.html) (German)
- [isri](https://www.nltk.org/api/nltk.stem.isri.html) (Arabic)
- [krovetz](https://dl.acm.org/doi/10.1145/160688.160718) (English)
- [lancaster](https://www.nltk.org/api/nltk.stem.lancaster.html) (English)
- [porter](https://www.nltk.org/api/nltk.stem.porter.html) (English)


## Tokenizers
[Tokenizers](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization) divide a string into smaller units, such as words.  
[retriv](https://github.com/AmenRa/retriv) supports the following tokenizers:
- [whitespace](https://www.nltk.org/api/nltk.tokenize.html)
- [word](https://www.nltk.org/api/nltk.tokenize.html)
- [wordpunct](https://www.nltk.org/api/nltk.tokenize.html)
- [sent](https://www.nltk.org/api/nltk.tokenize.html)


## Stop-word Lists
[retriv](https://github.com/AmenRa/retriv) supports [stop-word](https://en.wikipedia.org/wiki/Stop_word) lists for the following languages: Arabic, Azerbaijani, Basque, Bengali, Catalan, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hinglish, Hungarian, Indonesian, Italian, Kazakh, Nepali, Norwegian, Portuguese, Romanian, Russian, Slovene, Spanish, Swedish, Tajik, and Turkish.