import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import pdb

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))

    # range(ord("!"), ord("~")+1) -- range(33, 127)  -- 94
    # range(ord("¡"), ord("¬")+1) -- range(161, 173) -- 12
    # range(ord("®"), ord("ÿ")+1) -- range(174, 256) -- 82
    # (94 + 12 + 82) ==> 188 == len(bs)
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    # ==> len(bs) == 256
    # ==> len(cs) == 256
    # (Pdb) dict(zip(bs, cs))
    # {33: '!', 34: '"', 35: '#', 36: '$', 37: '%', 38: '&', 39: "'", 40: '(', 41: ')', 42: '*', 43: '+', 44: ',', 45: '-', 46: '.', 
    # 47: '/', 48: '0', 49: '1', 50: '2', 51: '3', 52: '4', 53: '5', 54: '6', 55: '7', 56: '8', 57: '9', 58: ':', 59: ';', 60: '<', 
    # 61: '=', 62: '>', 63: '?', 64: '@', 65: 'A', 66: 'B', 67: 'C', 68: 'D', 69: 'E', 70: 'F', 71: 'G', 72: 'H', 73: 'I', 74: 'J', 
    # 75: 'K', 76: 'L', 77: 'M', 78: 'N', 79: 'O', 80: 'P', 81: 'Q', 82: 'R', 83: 'S', 84: 'T', 85: 'U', 86: 'V', 87: 'W', 88: 'X', 
    # 89: 'Y', 90: 'Z', 91: '[', 92: '\\', 93: ']', 94: '^', 95: '_', 96: '`', 97: 'a', 98: 'b', 99: 'c', 100: 'd', 101: 'e', 
    # 102: 'f', 103: 'g', 104: 'h', 105: 'i', 106: 'j', 107: 'k', 108: 'l', 109: 'm', 110: 'n', 111: 'o', 112: 'p', 113: 'q', 
    # 114: 'r', 115: 's', 116: 't', 117: 'u', 118: 'v', 119: 'w', 120: 'x', 121: 'y', 122: 'z', 123: '{', 124: '|', 125: '}', 
    # 126: '~', 161: '¡', 162: '¢', 163: '£', 164: '¤', 165: '¥', 166: '¦', 167: '§', 168: '¨', 169: '©', 170: 'ª', 171: '«', 
    # 172: '¬', 174: '®', 175: '¯', 176: '°', 177: '±', 178: '²', 179: '³', 180: '´', 181: 'µ', 182: '¶', 183: '·', 184: '¸', 
    # 185: '¹', 186: 'º', 187: '»', 188: '¼', 189: '½', 190: '¾', 191: '¿', 192: 'À', 193: 'Á', 194: 'Â', 195: 'Ã', 196: 'Ä', 
    # 197: 'Å', 198: 'Æ', 199: 'Ç', 200: 'È', 201: 'É', 202: 'Ê', 203: 'Ë', 204: 'Ì', 205: 'Í', 206: 'Î', 207: 'Ï', 208: 'Ð', 
    # 209: 'Ñ', 210: 'Ò', 211: 'Ó', 212: 'Ô', 213: 'Õ', 214: 'Ö', 215: '×', 216: 'Ø', 217: 'Ù', 218: 'Ú', 219: 'Û', 220: 'Ü', 
    # 221: 'Ý', 222: 'Þ', 223: 'ß', 224: 'à', 225: 'á', 226: 'â', 227: 'ã', 228: 'ä', 229: 'å', 230: 'æ', 231: 'ç', 232: 'è', 
    # 233: 'é', 234: 'ê', 235: 'ë', 236: 'ì', 237: 'í', 238: 'î', 239: 'ï', 240: 'ð', 241: 'ñ', 242: 'ò', 243: 'ó', 244: 'ô', 
    # 245: 'õ', 246: 'ö', 247: '÷', 248: 'ø', 249: 'ù', 250: 'ú', 251: 'û', 252: 'ü', 253: 'ý', 254: 'þ', 255: 'ÿ', 
    # 0: 'Ā', 1: 'ā', 2: 'Ă', 3: 'ă', 4: 'Ą', 5: 'ą', 6: 'Ć', 7: 'ć', 8: 'Ĉ', 9: 'ĉ', 10: 'Ċ', 11: 'ċ', 12: 'Č', 13: 'č', 14: 'Ď', 
    # 15: 'ď', 16: 'Đ', 17: 'đ', 18: 'Ē', 19: 'ē', 20: 'Ĕ', 21: 'ĕ', 22: 'Ė', 23: 'ė', 24: 'Ę', 25: 'ę', 26: 'Ě', 27: 'ě', 28: 'Ĝ', 
    # 29: 'ĝ', 30: 'Ğ', 31: 'ğ', 32: 'Ġ', 127: 'ġ', 128: 'Ģ', 129: 'ģ', 130: 'Ĥ', 131: 'ĥ', 132: 'Ħ', 133: 'ħ', 134: 'Ĩ', 135: 'ĩ', 
    # 136: 'Ī', 137: 'ī', 138: 'Ĭ', 139: 'ĭ', 140: 'Į', 141: 'į', 142: 'İ', 143: 'ı', 144: 'Ĳ', 145: 'ĳ', 146: 'Ĵ', 147: 'ĵ', 
    # 148: 'Ķ', 149: 'ķ', 150: 'ĸ', 151: 'Ĺ', 152: 'ĺ', 153: 'Ļ', 154: 'ļ', 155: 'Ľ', 156: 'ľ', 157: 'Ŀ', 158: 'ŀ', 159: 'Ł', 
    # 160: 'ł', 173: 'Ń'}

    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    '''
    'a  b   c d' --> 'a b c d'
    '''
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        # (Pdb) type(merges) -- <class 'list'>
        # (Pdb) len(merges) -- 262146
        # (Pdb) merges[0] -- '"bpe_simple_vocab_16e6.txt#version: 0.2'
        # (Pdb) merges[1] -- 'i n'
        # (Pdb) merges[2] -- 't h'
        # (Pdb) merges[3] -- 'a n'
        # (Pdb) merges[-2] -- 'scare d'
        # (Pdb) merges[-1] -- ''

        merges = merges[1:49152-256-2+1] # 49152-256-2+1 --> 48895

        # (Pdb) len(merges) -- 48894
        # (Pdb) merges[0] -- 'i n'
        # (Pdb) merges[-1] -- 'jeky ll</w>'

        merges = [tuple(merge.split()) for merge in merges]
        # (Pdb) len(merges) -- 48894
        # (Pdb) merges[0] -- ('i', 'n')
        # (Pdb) merges[-1] -- ('jeky', 'll</w>')

        vocab = list(bytes_to_unicode().values())
        # vocab -- ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', 
        #     '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
        #     'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 
        #     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '¡', 
        #     '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹', 'º', 
        #     '»', '¼', '½', '¾', '¿', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï', 'Ð', 'Ñ', 'Ò', 
        #     'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 
        #     'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ', 'Ā', 'ā', 'Ă', 
        #     'ă', 'Ą', 'ą', 'Ć', 'ć', 'Ĉ', 'ĉ', 'Ċ', 'ċ', 'Č', 'č', 'Ď', 'ď', 'Đ', 'đ', 'Ē', 'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'ę', 'Ě', 
        #     'ě', 'Ĝ', 'ĝ', 'Ğ', 'ğ', 'Ġ', 'ġ', 'Ģ', 'ģ', 'Ĥ', 'ĥ', 'Ħ', 'ħ', 'Ĩ', 'ĩ', 'Ī', 'ī', 'Ĭ', 'ĭ', 'Į', 'į', 'İ', 'ı', 'Ĳ', 
        #     'ĳ', 'Ĵ', 'ĵ', 'Ķ', 'ķ', 'ĸ', 'Ĺ', 'ĺ', 'Ļ', 'ļ', 'Ľ', 'ľ', 'Ŀ', 'ŀ', 'Ł', 'ł', 'Ń']
        vocab = vocab + [v+'</w>' for v in vocab] # ==> len(vocab) -- 512
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        # ==> len(vocab) -- 49408, 512(basic vocab) + 49408(merges) + 2(star + end symbol)
        # (Pdb) vocab[-2] -- '<|startoftext|>'
        # (Pdb) vocab[-1] -- '<|endoftext|>'

        self.encoder = dict(zip(vocab, range(len(vocab)))) # -- {'words': 42, ...}
        self.decoder = {v: k for k, v in self.encoder.items()} # --> {42: 'words', ...}
        self.bpe_ranks = dict(zip(merges, range(len(merges)))) # {('jeky', 'll</w>'): 48893}

        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        # bpe_path = 'clip/bpe_simple_vocab_16e6.txt.gz'
        # len(self.byte_encoder) -- 256
        # len(vocab) -- 49408
        # self.cache -- {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}

    def bpe(self, token):
        # token -- 'hello'
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        # word --> ('h', 'e', 'l', 'l', 'o</w>')
        pairs = get_pairs(word)
        # pairs --> {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o</w>')}

        if not pairs:
            return token+'</w>' # 'a</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            # (Pdb) self.bpe_ranks.get(('l', 'o</w>'), float('inf')) -- 2470
            # (Pdb) self.bpe_ranks.get(('h', 'e'), float('inf')) -- 139
            # (Pdb) self.bpe_ranks.get(('l', 'l'), float('inf')) -- 1145
            # (Pdb) self.bpe_ranks.get(('e', 'l'), float('inf')) -- 32

            if bigram not in self.bpe_ranks:
                break

            first, second = bigram # ('e', 'l') -- 32 is best
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j]) # [i:j] --> [i, j) in math semantics
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # new_word -- ['h', 'el', 'l', 'o</w>']
            # pairs -- {('l', 'o</w>'), ('h', 'el'), ('el', 'l')}
            # new_word -- ['h', 'ell', 'o</w>']
            # pairs -- {('h', 'ell'), ('ell', 'o</w>')}
            # new_word -- ['h', 'ello</w>']
            # pairs -- {('h', 'ello</w>')}

            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        # text -- 'a diagram'
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text): # ==> ['a', 'diagram']
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))

            # self.encoder -- {..., 'diffuser</w>': 49400, ... }
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
            # self.encoder['a</w>'] -- 320

        # bpe_tokens -- [320, 22697]
        # self.decoder[320] -- 'a</w>'
        # self.decoder[3306] -- 'hello</w>'
        # self.decoder[22697] -- 'diagram</w>'

        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
