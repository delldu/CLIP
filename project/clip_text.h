/************************************************************************************
***
***	Copyright 2023 Dell(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, Fri 28 Jul 2023 12:36:25 PM CST
***
************************************************************************************/
#ifndef CLIP_TEXT_H
#define CLIP_TEXT_H

#include <iostream>
#include <vector>
#include <unordered_map>

struct CLIPText {
public:
    std::vector<uint32_t> encode(const std::string& text);
    std::string decode(const std::vector<uint32_t>& tokens);

private:
    std::string bpe(const std::string& token_word);

    std::unordered_map<std::string, std::string> m_cache {
        { "<|startoftext|>", "<|startoftext|>" },
        { "<|endoftext|>", "<|endoftext|>" }
    };
};
#endif // CLIP_TEXT_H
