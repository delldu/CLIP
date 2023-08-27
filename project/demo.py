# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 02 Aug 2023 06:43:47 AM CST
# ***
# ************************************************************************************/
#

import CLIP

CLIP.predict("images/CLIP*.png", "output")


test_segment = {}
test_segment["images/example0.png"] = ['racket']
test_segment["images/example1.png"] = ['butterfly']
test_segment["images/example2.png"] = ['chair']
test_segment["images/example3.png"] = ['shelf']
test_segment["images/example4.png"] = ['chair']
test_segment["images/example5.png"] = ['bird']
test_segment["images/example6.png"] = ['shark']
test_segment["images/example7.png"] = ['bed', 'pillow']
test_segment["images/example8.png"] = ['black car', 'white car']

CLIP.segment(test_segment, "output")
