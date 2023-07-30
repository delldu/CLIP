#/************************************************************************************
#***
#***	Copyright 2023 Dell(dellrunning@gmail.com), All Rights Reserved.
#***
#***	File Author: Dell, Fri 28 Jul 2023 12:36:58 PM CST
#***
#************************************************************************************/
#

TARGET := clip_token

SOURCE :=  \
	main.cpp

INCS	:=
	#  -I$(BUILD_DIR)/include
LIBS	:= 
	#-L$(BUILD_DIR)/lib -lstd... -lstdc++

DEFINES :=

CFLAGS := -O2 -fPIC -Wall -Wextra
LDFLAGS := -fPIC

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CFLAGS   := ${CFLAGS} ${DEFINES}
CXXFLAGS := ${CXXFLAGS} ${DEFINES}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: ${TARGET}

${TARGET}: ${OBJECT} codec.h clip_text.h
	${CXX} ${LDFLAGS} -o ${TARGET} ${OBJECT} ${LIBS}

install:
	${STRIP} ${TARGET}
	install -m 755 ${TARGET} ${INSTALL_DIR}/bin

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cpp
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

%.o : %.c
	${CC} ${CFLAGS} ${INCS} -c $< -o $@

#****************************************************************************
# Depend block
#****************************************************************************
depend:

clean:
	rm -f core ${OBJECT} ${TARGET}

format:
	clang-format -i --style=WebKit clip_text.cpp	
