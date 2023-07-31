/************************************************************************************
***
***	Copyright 2023 Dell(dellrunning@gmail.com), All Rights Reserved.
***
***	File Author: Dell, Fri 28 Jul 2023 12:36:25 PM CST
***
************************************************************************************/

#include <iostream>

#include "clip_text.h"

int main(int argc, char** argv)
{
	std::string text = "a diagram, hello world, good, man, women !";
    CLIPText* codec = new CLIPText();

    std::vector<uint32_t> tokens = codec->encode(text);
    std::cout << "Encode: " << text << std::endl;
    for (auto& t : tokens)
        std::cout << t << " ";
    std::cout << std::endl;

    std::cout << "Decode: " << std::endl;
    std::string words = codec->decode(tokens);
    std::cout << words << std::endl;

	// test();

    return 0;
}
