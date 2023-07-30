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

// using namespace std;
using convert_type = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_type, wchar_t> converter;

static std::map<uint32_t, std::string> clip_text_byte_encoder {
	{33, "!"},
	{174, "®"},
};

static std::map<std::string, uint32_t> clip_text_word_encoder {
	{"a</w>", 320},
	{"hello</w>", 3306},
	{"diagram</w>", 22697},
};


static std::map<std::string, uint32_t> clip_text_byte_decoder {
	{"!", 33},
	{"®", 174},
};

static std::map<uint32_t, std::string> clip_text_word_decoder {
	{320, "a</w>"},
	{3306, "hello</w>"},
	{22697, "diagram</w>"},
};


static wchar_t ord(std::wstring ws)
{
    for (wchar_t wc : ws)
    	return wc;

    return 0;
}


void test_unicode(std::string str)
{
	std::cout << "raw str:" << str << std::endl;
    std::wstring ws = converter.from_bytes(str);
    for (wchar_t wc : ws)
    	std::cout << "wchar_t: " << wc << "" << ", byte: " << converter.to_bytes(wc) << std::endl;

	// wchar_t: 97, byte: a
	// wchar_t: 98, byte: b
	// wchar_t: 99, byte: c
	// wchar_t: 44, byte: ,
	// wchar_t: 32, byte:  
	// wchar_t: 49, byte: 1
	// wchar_t: 50, byte: 2
	// wchar_t: 51, byte: 3

    // for (wchar_t wc :  L"®")
    // 	std::cout << "one wc: " << wc << "" << ", byte: " << converter.to_bytes(wc) << std::endl;
    // for (wchar_t wc : L"ÿ")
    // 	std::cout << "one wc: " << wc << "" << ", byte: " << converter.to_bytes(wc) << std::endl;
    // for (wchar_t wc : L"®ÿ")
    // 	std::cout << "two wc: " << wc << "" << ", byte: " << converter.to_bytes(wc) << std::endl;
}

// def get_pairs(word):
//     """Return set of symbol pairs in a word.
//     Word is represented as tuple of symbols (symbols being variable-length strings).
//     """
//     pairs = set()
//     prev_char = word[0]
//     for char in word[1:]:
//         pairs.add((prev_char, char))
//         prev_char = char

//     return pairs




std::vector<uint32_t> clip_text_encode(std::string str)
{
	std::vector<uint32_t> tokens;

	std::cout << "Encoding 1:" << str << " ..." << std::endl;
	// str.erase(remove(str.begin(), str.end(), ' '), str.end());
	// str = regex_replace(str, std::regex("\\s"),"");
	// remove_if(str.begin(), str.end(), isspace);

	transform(str.begin(),str.end(), str.begin(),::tolower);
	std::cout << "Encoding 2:" << str << " ..." << std::endl;


	tokens.push_back(320);
	tokens.push_back(1024);

	std::cout << "Encoding Results:" << std::endl;
	for (auto &t: tokens) {
		std::cout << t << std::endl;
	}


	return tokens;
}

std::string clip_text_decode(std::vector<uint32_t> tokens)
{
	std::string text("This is decode");

	std::cout << "Decoding ... " << std::endl;
	for (auto &t: tokens) {
		std::cout << t << std::endl;
	}

	std::cout << "Decoding Results:" << std::endl;
	std::cout << text << std::endl;

	return text;
}





int main(int argc, char **argv)
{
	std::cout << "Hello world !" << std::endl;
	// const Regex kCLIPRegex(
	//     "(?i)(<\\|startoftext\\|>|<\\|endoftext\\|>|\\'s|\\'t|\\'re|\\'ve|"
	//     "\\'m|\\'ll|\\'d|[\\pL]+|[\\pN]|[^\\s\\pL\\pN]+)");
    // self.pat = re.compile(r"""
    // 		<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|
    // 		'm|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
	// const std::string kWhitespaceString("</w>");
	// const std::unordered_set<std::string> kSpecialTokens{
	//     "<|startoftext|>",
	//     "<|endoftext|>"};



	std::string s1("Hello world, this is don't  refine  test for clip text !!! {1,2,3}");
	// std::regex kCLIPRegex(R"(\d+)");
	const std::regex kCLIPRegex(
	    R"(<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^s[:alpha:]][:digit:]]+)");
	const std::string kWhitespaceString(R"(</w>)");
	const std::unordered_set<std::string> kSpecialTokens{
		R"(<|startoftext|>)", R"(<|endoftext|>)"};

	std::cout << s1 << std::endl;

	std::sregex_iterator it(s1.begin(), s1.end(), kCLIPRegex), end;

	while(it != end) {
		// std::cout << "size: " << it->size() << std::endl;
		for(unsigned i = 0; i < it->size(); ++i) {
			std::cout << (*it)[i] << std::endl;
		}
		++it;
	}


	// test_unicode("abc, 123");


	auto tokens = clip_text_encode("Many spcae             ,abc, def, 123, 456, Big</W> Words, xxxx</w> !!!");

	auto text = clip_text_decode(tokens);

	std::cout << "clip_text_byte_encoder:" << std::endl;
	for (auto &x:clip_text_byte_encoder) {
		std::cout << x.first << " -- " << x.second << std::endl;
	}


	std::cout << "clip_text_word_encoder:" << std::endl;
	for (auto &x:clip_text_word_encoder) {
		std::cout << x.first << " -- " << x.second << std::endl;
	}

	return 0;
}

