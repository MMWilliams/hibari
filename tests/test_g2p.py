import pytest

from hibari.g2p import G2P
from hibari.g2p import numbers as N
from hibari.g2p.normalize import normalize


@pytest.fixture(scope="module")
def g2p():
    return G2P()


@pytest.fixture(scope="module")
def g2p_gb():
    return G2P(british=True)


def test_numbers():
    assert N.cardinal(1234) == "one thousand, two hundred and thirty-four"
    assert N.cardinal(1001) == "one thousand and one"
    assert N.ordinal(23) == "twenty-third"
    assert N.ordinal(101) == "one hundred and first"
    assert N.year(1905) == "nineteen oh-five"
    assert N.year(2010) == "twenty ten"
    assert N.year(2005) == "two thousand and five"
    assert N.decimal("3.140") == "three point one four"


@pytest.mark.parametrize("text, expected", [
    ("at 9:30am", "at nine thirty A.M."),
    ("on 10/12/2019", "on October twelfth, 2019"),
    ("5km away", "5 kilometers away"),
    ("1 kg", "1 kilogram"),
    ("20°C", "20 degrees Celsius"),
    ("Dr. Smith", "Dr Smith"),
    ("Prof. Smith", "Professor Smith"),
    ("Main St. is", "Main Street is"),
    ("St. Louis", "Saint Louis"),
    ("Henry VIII", "Henry the eighth"),
    ("Chapter IV", "Chapter 4"),
    ("$3.2 billion", "3.2 billion dollars"),
    ("¥500", "500 yen"),
    ("3/4 cup", "3 quarters cup"),
    ("pages 2-3", "pages 2 to 3"),
    ("Jan. 5th, 2020", "January fifth, 2020"),
    ("x = 5", "x equals 5"),
])
def test_normalize(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize("text, expected", [          # expected = misaki + spaCy reference output
    ("Hello there.", "həlˈO ðˈɛɹ."),
    ("The apple and the pear.", "ði ˈæpᵊl ænd ðə pˈɛɹ."),
    ("Please read it now; he read it yesterday.", "plˈiz ɹˈid ɪt nˈW; hi ɹˈɛd ɪt jˈɛstəɹdˌA."),
    ("They live in a live studio.", "ðˌA lˈɪv ɪn ɐ lˈIv stˈudiO."),
    ("He used to run.", "hˌi jˈust tə ɹˈʌn."),
    ("It costs $5.", "ˌɪt kˈɔsts fˈIv dˈɑləɹz."),
    ("In 1999, 10% of 1,200 people.", "ɪn nˌIntˈin nˈIndi nˈIn, tˈɛn pəɹsˈɛnt ʌv wˈʌn θˈWzᵊnd tˈu hˈʌndɹəd pˈipᵊl."),
    ("Come in!", "kˈʌm ˈɪn!"),
    ("To be or not to be, to eat.", "tə bi ɔɹ nˌɑt tə bˈi, tʊ ˈit."),
    ("She will record a new record on Tuesday.", "ʃˌi wɪl ɹəkˈɔɹd ɐ nˈu ɹˈɛkəɹd ˌɔn tˈuzdˌA."),
    ("The 2nd and 3rd of May, 2024.", "ðə sˈɛkənd ænd θˈɜɹd ʌv mˈA, twˈɛnti twˈɛnti fˈɔɹ."),
])
def test_phonemes(g2p, text, expected):
    assert g2p(text) == expected


def test_oov_and_custom_phonemes(g2p):
    ps = g2p("Blorptastic!")
    assert ps.endswith("!") and len(ps) > 5
    assert g2p("Say [hibari](/hibˈɑɹi/) now.") == "sˈA hibˈɑɹi nˈW."
    assert g2p("Say hibari now.") == "sˈA hɪbˈɑɹɹi nˈW."          # OOV net, same as espeak here


def test_empty_and_punctuation(g2p):
    assert g2p("") == ""
    assert g2p("...") == "..."


@pytest.mark.parametrize("text, expected", [          # expected = misaki british=True reference output
    ("Hello there.", "həlˈQ ðˈɛː."),
    ("The apple and the pear.", "ði ˈapᵊl and ðə pˈɛː."),
    ("He used to run; she washes the dishes and faded fast.", "hˌiː jˈuːst tə ɹˈʌn; ʃiː wˈɒʃɪz ðə dˈɪʃɪz and fˈAdɪd fˈɑːst."),
    ("Please read it now; he read it yesterday.", "plˈiːz ɹˈiːd ɪt nˈW; hiː ɹˈɛd ɪt jˈɛstədA."),
    ("It costs £5.50 and $3.", "ˌɪt kˈɒsts fˈIv pˈWndz and fˈɪfti pˈɛns and θɹˈiː dˈɒləz."),
    ("A car, a bar, better water.", "ɐ kˈɑː, ɐ bˈɑː, bˈɛtə wˈɔːtə."),
    ("They were staring and starring.", "ðˌA wɜː stˈɛːɹɪŋ and stˈɑːɹɪŋ."),
])
def test_british_phonemes(g2p_gb, text, expected):
    assert g2p_gb(text) == expected


def test_british_differs_from_american(g2p, g2p_gb):
    assert g2p("Better water.") == "bˈɛTəɹ wˈɔTəɹ." and g2p_gb("Better water.") == "bˈɛtə wˈɔːtə."
    ps = g2p_gb("Blorptastic!")                                      # British OOV net: non-rhotic, long vowels
    assert ps.endswith("!") and len(ps) > 5 and "ɹ" not in ps.split("ˈ")[0]
