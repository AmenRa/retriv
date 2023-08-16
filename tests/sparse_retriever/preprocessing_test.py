import pytest

from retriv.sparse_retriever.preprocessing import multi_preprocessing, preprocessing
from retriv.sparse_retriever.preprocessing.stemmer import get_stemmer
from retriv.sparse_retriever.preprocessing.stopwords import get_stopwords
from retriv.sparse_retriever.preprocessing.tokenizer import get_tokenizer


# FIXTURES =====================================================================
@pytest.fixture
def stemmer():
    return get_stemmer("english")


@pytest.fixture
def stopwords():
    return get_stopwords("english")


@pytest.fixture
def tokenizer():
    return get_tokenizer("whitespace")


@pytest.fixture
def docs():
    return """Black Sabbath were an English rock band formed in Birmingham in 1968 by guitarist Tony Iommi, drummer Bill Ward, bassist Geezer Butler and vocalist Ozzy Osbourne. They are often cited as pioneers of heavy metal music. The band helped define the genre with releases such as Black Sabbath (1970), Paranoid (1970) and Master of Reality (1971). The band had multiple line-up changes following Osbourne's departure in 1979 and Iommi is the only constant member throughout their history. 
    After previous iterations of the group - the Polka Tulk Blues Band and Earth - the band settled on the name Black Sabbath in 1969. They distinguished themselves through occult themes with horror-inspired lyrics and down-tuned guitars. Signing to Philips Records in November 1969, they released their first single, "Evil Woman", in January 1970, and their debut album, Black Sabbath, was released the following month. Though it received a negative critical response, the album was a commercial success, leading to a follow-up record, Paranoid, later that year. The band's popularity grew, and by 1973's Sabbath Bloody Sabbath, critics were starting to respond favourably.
    Osbourne's excessive substance abuse led to his firing in 1979. He was replaced by former Rainbow vocalist Ronnie James Dio. Following two albums with Dio, Black Sabbath endured many personnel changes in the 1980s and 1990s that included vocalists Ian Gillan, Glenn Hughes, Ray Gillen and Tony Martin, as well as several drummers and bassists. Martin, who replaced Gillen in 1987, was the second-longest serving vocalist and recorded three albums with Black Sabbath before his dismissal in 1991. That same year, Iommi and Butler were rejoined by Dio and drummer Vinny Appice to record Dehumanizer (1992). After two more studio albums with Martin, who replaced Dio in 1993, the band's original line-up reunited in 1997 and released a live album, Reunion, the following year; they continued to tour occasionally until 2005. Other than various back catalogue reissues and compilation albums, as well as the Mob Rules-era line-up reunited as Heaven & Hell, there was no further activity under the Black Sabbath name for six years. They reunited in 2011 and released their final studio album and 19th overall, 13, in 2013, which features all of the original members except Ward. During their farewell tour, the band played their final concert in their home city of Birmingham on 4 February 2017. Occasional partial reunions have happened since, most recently when Osbourne and Iommi performed together at the closing ceremony of the 2022 Commonwealth Games in Birmingham.
    Black Sabbath have sold over 70 million records worldwide as of 2013, making them one of the most commercially successful heavy metal bands. Black Sabbath, together with Deep Purple and Led Zeppelin, have been referred to as the "unholy trinity of British hard rock and heavy metal in the early to mid-seventies". They were ranked by MTV as the "Greatest Metal Band of All Time" and placed second on VH1's "100 Greatest Artists of Hard Rock" list. Rolling Stone magazine ranked them number 85 on their "100 Greatest Artists of All Time". Black Sabbath were inducted into the UK Music Hall of Fame in 2005 and the Rock and Roll Hall of Fame in 2006. They have also won two Grammy Awards for Best Metal Performance, and in 2019 the band were presented a Grammy Lifetime Achievement Award.""".split(
        "\n"
    )


# TESTS ========================================================================
def test_multi_preprocessing(docs, stemmer, stopwords, tokenizer):
    out = [
        preprocessing(
            doc,
            stemmer=stemmer,
            stopwords=stopwords,
            tokenizer=tokenizer,
            do_lowercasing=True,
            do_ampersand_normalization=True,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
        )
        for doc in docs
    ]
    multi_out = list(
        multi_preprocessing(
            docs,
            stemmer=stemmer,
            stopwords=stopwords,
            tokenizer=tokenizer,
            n_threads=4,
            do_lowercasing=True,
            do_ampersand_normalization=True,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
        )
    )
    assert len(out) == len(multi_out)
    assert out[0] == multi_out[0]
    assert out[1] == multi_out[1]
    assert out[2] == multi_out[2]
    assert out[3] == multi_out[3]
    assert out == multi_out
