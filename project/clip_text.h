/************************************************************************************
***
***	Copyright 2023 Dell(dellrunning@gmail.com), All Rights Reserved.
***
***	File Author: Dell, Fri 28 Jul 2023 12:36:25 PM CST
***
************************************************************************************/
#ifndef CLIP_TEXT_H
#define CLIP_TEXT_H

#include <algorithm>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <unordered_map>
#include <unordered_set>


struct CLIPText {
public:
    std::vector<uint32_t> encode(const std::string& text);
    std::string decode(const std::vector<uint32_t>& tokens);
    std::vector<std::string> bpe(const std::string& token);

private:
	std::map<std::string, std::string> m_cache {
		{"<|startoftext|>": "<|startoftext|>"},
		{"<|endoftext|>": "<|endoftext|>"}
	};
};
#endif // CLIP_TEXT_H


/************************************************************************************
***
***  CLIP Text Implementation
***
************************************************************************************/


#ifdef CLIP_TEXT_IMPLEMENTATION

#include "codec.h"

#define BPE_SEPERATOR " " # should match seperator of clip_text_bpe_ranks key 


static std::vector<std::string> get_clip_words(std::string str)
{
    const std::regex kCLIPRegex(
        R"(<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]+|[^s[:alpha:]]+[:digit:]]+)");
    // const std::regex kWord(R"(</w>|</W>)");

    std::vector<std::string> words;

    // str = std::regex_replace(str, kWord, ""); // remove '</w>'

    std::sregex_iterator it(str.begin(), str.end(), kCLIPRegex), end;
    while(it != end) {
        for(unsigned i = 0; i < it->size(); ++i) {
            std::string temp = (*it)[i];

            remove_if(temp.begin(), temp.end(), isspace);

            if (temp.length() > 0)
                words.push_back(temp);
        }
        ++it;
    }

    return words;
}


std::pair<std::string, std::string> split_tokens(std::string s, std::string delimiter)
{
    auto pos = s.find(delimiter);
    return std::make_pair(s.substr(0, pos), s.substr(pos + delimiter.length()));
}

int list_str_index(std::vector<std::string> list, std::string element, int start)
{
    // Equivalent to: list.index(element, start)
    for (std::size_t i = start; i < list.size(); ++i) {
        if (list[i] == element)
            return i;
    }
    return -1;
}


std::vector<std::string> get_pairs(std::vector<std::string> token_list)
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

uint32_t GetBPEMergeRank_(std::string pair)
{
    if (clip_text_bpe_ranks.find(pair) != clip_text_bpe_ranks.end()) {
        return clip_text_bpe_ranks.at(pair);
    }
    return 0xfffffffe;// enough than vocab size
}

std::string FindBestPair_(std::vector<std::string> pairs)
{
    // Equivalent to:
    //    min(pairs, key = lambda pair: self.bpe_merge_ranks.get(pair,
    //    float('inf')))
    uint32_t best_pair_idx = 0;
    uint32_t best_rank = GetBPEMergeRank_(pairs[best_pair_idx]);

    for (std::size_t i = 1; i < pairs.size(); ++i) {
        uint32_t rank = GetBPEMergeRank_(pairs[i]);
        if (rank < best_rank) {
            best_pair_idx = i;
            best_rank = rank;
        }
    }
    return pairs[best_pair_idx];
}



std::vector<uint32_t> CLIPText::encode(const std::string& text)
{
	std::vector<uint32_t> bpe_tokens;

    std::vector<std::string> words = get_clip_words(text);

    for (std::string word: words) {
        std::cout << w << std::endl;
        std::string token_word;
        for (size_t i = 0; i < word.size(); i++)
            token_word.push_back(clip_text_byte_encoder[(uint32_t)word.at(i)]);

        bpe_word = bpe(token_word);
        bpe_tokens.push_back(clip_text_word_encoder[bpe_word]);
    }

	return bpe_tokens;
}


std::string CLIPText::bpe(const std::string& token_word)
{
	std::string s, bpe_word;

    if (m_cache.find(token_word) != m_cache.end())
        return m_cache[token_word];

    std::vector<std::string> token_list;
    for (size_t i = 0; i < token_word.size(); i++) {
        s = token_word.at(i);
        if (i == token_word.size() - 1)
            s += "</w>";
        token_list.push_back(s);
    }
    // example "hello" --> ("h", "e", "l", "l", "o</w>")

    std::vector<std::string> pairs = get_pairs(token_list);
    if (pairs.empty())
        return { token_word + "</w>" };


    while (true) {
        auto bigram = FindBestPair_(pairs); // bi-gram
        if (clip_text_bpe_ranks.find(bigram) == clip_text_bpe_ranks.end())
            break;

        auto parts = split_tokens(bigram, BPE_SEPERATOR);
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
            pairs = get_pairs(token_list);
        }
    }

    bpe_word = token_list.at(0)
    m_cache.insert(token_word, bpe_word);

	return bpe_word;	
}


// text = ''.join([self.decoder[token] for token in tokens])
// text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
// return text
std::string CLIPText::decode(const std::vector<uint32_t>& tokens)
{
    const std::regex kWord(R"(</w>|</W>)");

	std::string text="";

    std::string bpe_text="";
    for (size_t i = 0; i < tokens.size(); i++) {
        bpe_text += clip_text_word_decoder[(uint32_t)i];
    }

    for (size_t i = 0; i < bpe_text.size(); i++) {
        text += clip_text_byte_encoder[(uint32_t)bpe_text.at(i)];
    }

    text = std::regex_replace(text, kWord, " "); // remove '</w>'

	return text;
}

#endif // CLIP_TEXT_IMPLEMENTATION