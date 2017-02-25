IF(NOT MATOG_FOUND)
	SET(MATOG_FOUND "NO")

	IF(NOT MATOG_DIR)
		SET(MATOG_DIR $ENV{MATOG_DIR} CACHE PATH "C:/joemuell/matog/install")
	ENDIF()

	IF(NOT MATOG_DIR)
		MESSAGE("Please set the environment variable MATOG_DIR to the path of MATOG.")
	ENDIF()

	# search for headers
	FIND_PATH(MATOG_INCLUDE_DIR "Matog.h" PATHS ${MATOG_DIR} PATH_SUFFIXES "include")

	# search for executables
	FIND_PROGRAM(MATOG_BINARY "matog-code" PATHS ${MATOG_DIR} PATH_SUFFIXES "bin")

	# search for libraries
	FIND_LIBRARY(MATOG_LIBRARY					"matog-lib"				PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_LIBRARY_DEBUG 			"matog-lib_debug"		PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_LOG_LIBRARY				"matog-log"				PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_LOG_LIBRARY_DEBUG 		"matog-log_debug"		PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_UTIL_LIBRARY       		"matog-util"			PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_UTIL_LIBRARY_DEBUG		"matog-util_debug"		PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	#FIND_LIBRARY(MATOG_PUGIXML_LIBRARY			"pugixml"				PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	#FIND_LIBRARY(MATOG_PUGIXML_LIBRARY_DEBUG	"pugixml_debug"			PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_SQLITE_LIBRARY			"sqlite3"				PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_SQLITE_LIBRARY_DEBUG		"sqlite3_debug"			PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_CTEMPLATE_LIBRARY		"ctemplate"				PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_CTEMPLATE_LIBRARY_DEBUG	"ctemplate_debug"		PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_JSONCPP_LIBRARY			"jsoncpp"				PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")
	FIND_LIBRARY(MATOG_JSONCPP_LIBRARY_DEBUG	"jsoncpp_debug"			PATHS ${MATOG_DIR} PATH_SUFFIXES "lib")

	IF(
		MATOG_INCLUDE_DIR AND 
		MATOG_BINARY AND 
		MATOG_DIR AND
		MATOG_LIBRARY AND 
		MATOG_LIBRARY_DEBUG AND 
		MATOG_LOG_LIBRARY AND 
		MATOG_LOG_LIBRARY_DEBUG AND 
		MATOG_UTIL_LIBRARY AND 
		MATOG_UTIL_LIBRARY_DEBUG AND 
		#MATOG_PUGIXML_LIBRARY AND 
		#MATOG_PUGIXML_LIBRARY_DEBUG AND 
		MATOG_SQLITE_LIBRARY AND 
		MATOG_SQLITE_LIBRARY_DEBUG AND 
		MATOG_CTEMPLATE_LIBRARY AND 
		MATOG_CTEMPLATE_LIBRARY_DEBUG AND
		MATOG_JSONCPP_LIBRARY AND
		MATOG_JSONCPP_LIBRARY_DEBUG
	)
		SET(MATOG_FOUND "YES")
	
		SET(MATOG_LIBRARIES 
			optimized 	${MATOG_LIBRARY} 
			debug 		${MATOG_LIBRARY_DEBUG} 
			optimized 	${MATOG_LOG_LIBRARY} 
			debug 		${MATOG_LOG_LIBRARY_DEBUG} 
			optimized 	${MATOG_UTIL_LIBRARY} 
			debug 		${MATOG_UTIL_LIBRARY_DEBUG} 
			#optimized 	${MATOG_PUGIXML_LIBRARY} 
			#debug 		${MATOG_PUGIXML_LIBRARY_DEBUG} 
			optimized 	${MATOG_SQLITE_LIBRARY} 
			debug 		${MATOG_SQLITE_LIBRARY_DEBUG} 
			optimized 	${MATOG_CTEMPLATE_LIBRARY} 
			debug 		${MATOG_CTEMPLATE_LIBRARY_DEBUG}
			optimized	${MATOG_JSONCPP_LIBRARY}
			debug		${MATOG_JSONCPP_LIBRARY_DEBUG}
			${CUDA_cupti_LIBRARY}
		)
	
		MARK_AS_ADVANCED(
			${MATOG_LIBRARIES} 
			${MATOG_INCLUDE_DIR} 
			${MATOG_BINARY} 
			${MATOG_LIBRARY} 
			${MATOG_LIBRARY_DEBUG} 
			${MATOG_LOG_LIBRARY} 
			${MATOG_LOG_LIBRARY_DEBUG} 
			${MATOG_UTIL_LIBRARY} 
			${MATOG_UTIL_LIBRARY_DEBUG} 
			#${MATOG_PUGIXML_LIBRARY} 
			#${MATOG_PUGIXML_LIBRARY_DEBUG} 
			${MATOG_SQLITE_LIBRARY} 
			${MATOG_SQLITE_LIBRARY_DEBUG} 
			${MATOG_CTEMPLATE_LIBRARY} 
			${MATOG_CTEMPLATE_LIBRARY_DEBUG} 
			${MATOG_JSONCPP_LIBRARY}
			${MATOG_JSONCPP_LIBRARY_DEBUG}
		)	
	
		MESSAGE(STATUS "Found MATOG")
	ELSE()
		MESSAGE(STATUS "MATOG not found")
		MESSAGE(STATUS "MATOG_DIR                     = ${MATOG_DIR}")
		MESSAGE(STATUS "MATOG_INCLUDE_DIR             = ${MATOG_INCLUDE_DIR}")
		MESSAGE(STATUS "MATOG_BINARY                  = ${MATOG_BINARY}")
		MESSAGE(STATUS "MATOG_LIBRARY                 = ${MATOG_LIBRARY}")
		MESSAGE(STATUS "MATOG_LIBRARY_DEBUG           = ${MATOG_LIBRARY_DEBUG}") 
		MESSAGE(STATUS "MATOG_LOG_LIBRARY             = ${MATOG_LOG_LIBRARY}")
		MESSAGE(STATUS "MATOG_LOG_LIBRARY_DEBUG       = ${MATOG_LOG_LIBRARY_DEBUG}")
		MESSAGE(STATUS "MATOG_UTIL_LIBRARY            = ${MATOG_UTIL_LIBRARY}")
		MESSAGE(STATUS "MATOG_UTIL_LIBRARY_DEBUG      = ${MATOG_UTIL_LIBRARY_DEBUG}")
		MESSAGE(STATUS "MATOG_PUGIXML_LIBRARY         = ${MATOG_PUGIXML_LIBRARY}")
		MESSAGE(STATUS "MATOG_PUGIXML_LIBRARY_DEBUG   = ${MATOG_PUGIXML_LIBRARY_DEBUG}")
		MESSAGE(STATUS "MATOG_SQLITE_LIBRARY          = ${MATOG_SQLITE_LIBRARY}")
		MESSAGE(STATUS "MATOG_SQLITE_LIBRARY_DEBUG    = ${MATOG_SQLITE_LIBRARY_DEBUG}")
		MESSAGE(STATUS "MATOG_CTEMPLATE_LIBRARY       = ${MATOG_CTEMPLATE_LIBRARY}")
		MESSAGE(STATUS "MATOG_CTEMPLATE_LIBRARY_DEBUG = ${MATOG_CTEMPLATE_LIBRARY_DEBUG}")
		MESSAGE(STATUS "MATOG_JSONCPP_LIBRARY         = ${MATOG_JSONCPP_LIBRARY}")
		MESSAGE(STATUS "MATOG_JSONCPP_LIBRARY_DEBUG   = ${MATOG_JSONCPP_LIBRARY_DEBUG}")
	ENDIF()

	IF(MATOG_FIND_REQUIRED AND NOT MATOG_FOUND)
		MESSAGE(FATAL_ERROR "Could NOT find MATOG.")
	ENDIF()

	INCLUDE_DIRECTORIES(${MATOG_INCLUDE_DIR})
ENDIF()