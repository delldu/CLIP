/************************************************************************************
***
***	Copyright 2023 Dell(dellrunning@gmail.com), All Rights Reserved.
***
***	File Author: Dell, Fri 28 Jul 2023 12:36:25 PM CST
***
************************************************************************************/

#include <iostream>
#include <cstring>

#include "clip_text.h"

// head -n 48896 bpe_simple_vocab_16e6.txt | grep -v "bpe_simple_vocab_16e6" > /tmp/bpe_vocab.txt
// xxd -i /tmp/bpe_vocab.txt bpe_vocab.h
// unsigned char _tmp_bpe_vocab_txt, _tmp_bpe_vocab_txt_len=524605

// #define BPE_VOCAB_SIZE 48895

// #include "bpe_vocab.h"

// void test()
// {

// 	uint32_t n = 0;
// 	char *v[BPE_VOCAB_SIZE], *s, *p;

// 	v[n++] = s = p = (char *) _tmp_bpe_vocab_txt;
// 	for (; *p && n < BPE_VOCAB_SIZE; ++p) {
// 		if (*p == '\n') {
// 			*p = '\0';
// 			v[n++] = s;
// 			s = p + 1;
// 		}
// 	}
// 	for (uint32_t i = 0; i < BPE_VOCAB_SIZE; i++) {
// 		p = strchr(v[i], ' ');
// 		if (p)
// 			*p = '\0';

// 		std::cout << i << ": " << v[i] << std::endl;
// 	}
// }


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
