/************************************************************************************
***
*** Copyright 2023 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Fri 28 Jul 2023 12:36:25 PM CST
***
************************************************************************************/

#include "clip_token.h"

#include <regex>
#include <unordered_set>
#include <unordered_map>

#define BPE_SEPERATOR " " // same as seperator of bpe ranks key
#define ARRAY_SIZE(x) (int)(sizeof(x) / sizeof(x[0]))

#include "clip_codec.h" // including codec_byte_list, codec_word_list, bpe_ranks_list

static std::vector<std::string> get_clip_words(std::string str)
{
    // const Regex kCLIPRegex(
    //     "(?i)(<\\|startoftext\\|>|<\\|endoftext\\|>|\\'s|\\'t|\\'re|\\'ve|"
    //     "\\'m|\\'ll|\\'d|[\\pL]+|[\\pN]|[^\\s\\pL\\pN]+)");

    const std::regex kCLIPRegex(
        R"(<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]+|[^s[:alpha:]]+[:digit:]]+)");

    std::vector<std::string> words;

    std::sregex_iterator it(str.begin(), str.end(), kCLIPRegex), end;
    while (it != end) {
        for (size_t i = 0; i < it->size(); ++i) {
            std::string temp = (*it)[i];

            remove_if(temp.begin(), temp.end(), isspace);

            if (temp.length() > 0)
                words.push_back(temp);
        }
        ++it;
    }

    return words;
}

static std::pair<std::string, std::string> split_bpe_pair(std::string s)
{
    std::string delimiter = BPE_SEPERATOR;

    auto pos = s.find(delimiter);
    return std::make_pair(s.substr(0, pos), s.substr(pos + delimiter.length()));
}

static int list_str_index(std::vector<std::string> list, std::string element, int start)
{
    for (std::size_t i = start; i < list.size(); ++i) {
        if (list[i] == element)
            return i;
    }
    return -1;
}

static std::vector<std::string> get_bpe_pairs(std::vector<std::string> token_list)
{
    std::vector<std::string> pairs_vec;
    std::unordered_set<std::string> pairs;

    if (token_list.empty())
        return pairs_vec;

    std::string prev_token = token_list[0];
    for (std::size_t i = 1; i < token_list.size(); ++i) {
        pairs.insert(prev_token + BPE_SEPERATOR + token_list[i]);
        prev_token = token_list[i];
    }
    pairs_vec.insert(pairs_vec.end(), pairs.begin(), pairs.end());
    return pairs_vec;
}

uint32_t CLIPToken::get_bpe_rank_index(std::string pair)
{
    if (m_bpe_ranks.find(pair) != m_bpe_ranks.end()) {
        return m_bpe_ranks.at(pair);
    }
    return 0xfffffffe; // one const more than vocab size
}

std::string CLIPToken::find_best_bpe_pair(std::vector<std::string> pairs)
{
    uint32_t best_pair_index = 0;
    uint32_t best_rank = get_bpe_rank_index(pairs[best_pair_index]);

    for (std::size_t i = 1; i < pairs.size(); ++i) {
        uint32_t rank = get_bpe_rank_index(pairs[i]);
        if (rank < best_rank) {
            best_pair_index = i;
            best_rank = rank;
        }
    }
    return pairs[best_pair_index];
}

CLIPToken::CLIPToken()
{
    // std::unordered_map<uint32_t, std::string> m_byte_encoder;
    // std::unordered_map<std::string, uint32_t> m_word_encoder;
    // std::unordered_map<std::string, uint32_t> m_bpe_ranks;

    std::string s;
    for (uint32_t i = 0; i < ARRAY_SIZE(codec_byte_list); i++) {
        s = codec_byte_list[i];
        m_byte_encoder[i] = s;
    }

    for (uint32_t i = 0; i < ARRAY_SIZE(codec_word_list); i++) {
        s = codec_word_list[i];
        m_word_encoder[s] = i;
    }

    for (uint32_t i = 0; i < ARRAY_SIZE(bpe_ranks_list); i++) {
        s = bpe_ranks_list[i];
        m_bpe_ranks[s] = i;
    }

    for (auto &x:m_byte_encoder)
        m_byte_decoder[x.second] = x.first;

    for (auto &x:m_word_encoder)
        m_word_decoder[x.second] = x.first;
}

std::vector<uint32_t> CLIPToken::encode(const std::string& text)
{
    std::vector<uint32_t> bpe_tokens;

    std::vector<std::string> words = get_clip_words(text);
    // example: "a digram" ==> ["a", "diagram"]

    for (std::string word : words) {
        // convert normal word to bpe word
        std::string bpe_word = "";
        for (size_t i = 0; i < word.size(); i++) {
            bpe_word += m_byte_encoder[(uint32_t)word.at(i)]; // m_byte_encoder[0..255], so here is safe
        }

        std::string best_match_word = bpe(bpe_word);

        // convert bpe word to token
        if (m_word_encoder.find(best_match_word) != m_word_encoder.end()) {
            bpe_tokens.push_back(m_word_encoder[best_match_word]);
        }
    }

    return bpe_tokens;
}

std::string CLIPToken::bpe(const std::string& bpe_word)
{
    std::string s, best_match_word;

    if (m_cache.find(bpe_word) != m_cache.end())
        return m_cache[bpe_word];

    std::vector<std::string> token_list;
    for (size_t i = 0; i < bpe_word.size(); i++) {
        s = bpe_word.at(i);
        if (i == bpe_word.size() - 1)
            s += "</w>";
        token_list.push_back(s);
    }
    // example "hello" --> ("h", "e", "l", "l", "o</w>")

    std::vector<std::string> pairs = get_bpe_pairs(token_list);
    // ("he", "el", "ll", "lo</w>")

    if (pairs.empty())
        return { bpe_word + "</w>" };

    while (true) {
        std::string bigram = find_best_bpe_pair(pairs); // bi-gram
        if (m_bpe_ranks.find(bigram) == m_bpe_ranks.end())
            break;

        auto parts = split_bpe_pair(bigram);
        std::vector<std::string> new_token_list;
        std::size_t i = 0;
        while (i < token_list.size()) {
            auto j = list_str_index(token_list, parts.first, i);
            if (j != -1) {
                for (int k = i; k < j; k++)
                    new_token_list.push_back(token_list[k]);
                i = j;
            } else {
                for (std::size_t k = i; k < token_list.size(); k++)
                    new_token_list.push_back(token_list[k]);
                break;
            }

            if (token_list[i] == parts.first && i < (token_list.size() - 1) && token_list[i + 1] == parts.second) {
                new_token_list.push_back(parts.first + parts.second);
                i += 2;
            } else {
                new_token_list.push_back(token_list[i]);
                i += 1;
            }
        }

        token_list = new_token_list;
        if (token_list.size() == 1) {
            break;
        } else {
            pairs = get_bpe_pairs(token_list);
        }
    }

    // token_list.size() must be 0 or 1
    if (token_list.size() == 1) {
        best_match_word = token_list.at(0);
        m_cache[bpe_word] = best_match_word;
    }

    return best_match_word;
}

std::string CLIPToken::decode(const std::vector<uint32_t>& tokens)
{
    const std::regex kWord(R"(</w>|</W>)");
    std::string s;

    std::string text = "";

    // convert tokens to bpe words
    std::string bpe_words = "";
    for (size_t i = 0; i < tokens.size(); i++) {
        uint32_t t = tokens.at(i);
        if (m_word_decoder.find(t) != m_word_decoder.end()) {
            bpe_words += m_word_decoder[t];
        }
    }

    // covert bpe words to normal string
    for (size_t i = 0; i < bpe_words.size(); i++) {
        s = bpe_words.at(i);

        if (m_byte_decoder.find(s) != m_byte_decoder.end()) {
            text += m_byte_decoder[s];
        }
    }

    text = std::regex_replace(text, kWord, " "); // remove '</w>'

    return text;
}
