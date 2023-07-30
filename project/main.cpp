/************************************************************************************
***
***	Copyright 2023 Dell(dellrunning@gmail.com), All Rights Reserved.
***
***	File Author: Dell, Fri 28 Jul 2023 12:36:25 PM CST
***
************************************************************************************/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <unordered_set>
#include <codecvt>
#include <unordered_map>

#define CLIP_TEXT_IMPLEMENTATION
#include "clip_text.h"

// std::vector<uint32_t> clip_text_encode(std::string str)
// {
// 	std::vector<uint32_t> tokens;

// 	std::cout << "Encoding 1:" << str << " ..." << std::endl;
// 	// str.erase(remove(str.begin(), str.end(), ' '), str.end());
// 	// str = regex_replace(str, std::regex("\\s"),"");
// 	// remove_if(str.begin(), str.end(), isspace);

// 	transform(str.begin(),str.end(), str.begin(),::tolower);
// 	std::cout << "Encoding 2:" << str << " ..." << std::endl;


// 	tokens.push_back(320);
// 	tokens.push_back(1024);

// 	std::cout << "Encoding Results:" << std::endl;
// 	for (auto &t: tokens) {
// 		std::cout << t << std::endl;
// 	}


// 	return tokens;
// }

// std::string clip_text_decode(std::vector<uint32_t> tokens)
// {
// 	std::string text("This is decode");

// 	std::cout << "Decoding ... " << std::endl;
// 	for (auto &t: tokens) {
// 		std::cout << t << std::endl;
// 	}

// 	std::cout << "Decoding Results:" << std::endl;
// 	std::cout << text << std::endl;

// 	return text;
// }


int main(int argc, char **argv)
{
	std::cout << "Hello world !" << std::endl;

	CLIPText *codec = new CLIPText();

	std::vector<uint32_t> tokens = codec->encode("a diagram");
	for (auto &t: tokens) {
		std::cout << "Encode: " << t << std::endl;
	}

	std::string words = codec->decode(tokens);
	std::cout << "Decode: " << words << std::endl;

	return 0;
}

