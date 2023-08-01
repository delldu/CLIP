/************************************************************************************
***
***	Copyright 2023 Dell(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, Fri 28 Jul 2023 12:36:25 PM CST
***
************************************************************************************/
#ifndef CLIP_TOKEN_H
#define CLIP_TOKEN_H

#include <iostream>
#include <vector>
#include <unordered_map>


struct CLIPToken {
public:
    CLIPToken();

    std::vector<uint32_t> encode(const std::string& text);
    std::string decode(const std::vector<uint32_t>& tokens);

private:
    std::string bpe(const std::string& token_word);
    uint32_t get_bpe_rank_index(std::string pair);
    std::string find_best_bpe_pair(std::vector<std::string> pairs);

    std::unordered_map<uint32_t, std::string> m_byte_encoder;
    std::unordered_map<std::string, uint32_t> m_word_encoder;
    std::unordered_map<std::string, uint32_t> m_bpe_ranks;

    std::unordered_map<std::string, uint32_t> m_byte_decoder;
    std::unordered_map<uint32_t, std::string> m_word_decoder;

    std::unordered_map<std::string, std::string> m_cache {
        { "<|startoftext|>", "<|startoftext|>" },
        { "<|endoftext|>", "<|endoftext|>" }
    };
};
#endif // CLIP_TOKEN_H
