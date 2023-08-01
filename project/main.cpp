/************************************************************************************
***
***	Copyright 2023 Dell(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, Fri 28 Jul 2023 12:36:25 PM CST
***
************************************************************************************/

#include <iostream>

#include "clip_token.h"

int main(int argc, char** argv)
{
    std::string text = "a diagram, hello world, good, man, women !";
    CLIPToken* codec = new CLIPToken();

    std::vector<uint32_t> tokens = codec->encode(text);
    std::cout << "Encode: " << text << std::endl;
    for (auto& t : tokens)
        std::cout << t << " ";
    std::cout << std::endl;

    std::cout << "Decode: " << std::endl;
    std::string words = codec->decode(tokens);
    std::cout << words << std::endl;

    // Encode: a diagram, hello world, good, man, women !
    // 320 22697 3306 1002 886 786 1507
    // Decode:
    // a diagram hello world good man women

    return 0;
}
